"""Study 27a: exact verification (engine/study27a/SPEC_study27a.md, frozen before this script was run). Checks ZA1-ZA13 as specified there.
Modules A1 (local-to-outer LP), A2 (attribution controls), A3 (inspection ranking), A4 (relay and masking), graph controls, clock invariance.
"""
import os, sys, json, itertools, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from scipy.optimize import linprog
from pce import outerrelay as O, brokerage as BR, braidknot as BK, masking as MK, stranddefect as SD, raagbraid as RG
W = O.W
results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(27); P = W.truth(); NS = O.NS
DEV = [c for c, _, _ in W.CORE_ALLOC]; MCY = ("M", "C", "Y"); RHY = ("R", "H", "Y")
M1, M0, Y1 = O.ev(lambda v: v["M"] == 1), O.ev(lambda v: v["M"] == 0), O.ev(lambda v: v["Y"] == 1)
g = Y1 * M1 / (M1 @ P) - Y1 * M0 / (M0 @ P); gstar = g @ P
h_ctx = [c for c in DEV if c != MCY]
INSP = [("M", "Y"), ("M", "R", "Y"), ("M", "H", "Y"), ("M", "C", "R", "Y"), ("C", "Y"), MCY, ("C", "R", "Y")]
TAUS = [gstar + 0.02, gstar + 0.04, gstar + 0.08]; ANCH = [[], [("M", "R", "Y")], [("M", "H", "Y")]]

# ------------------------------------------------------------------ generic helpers (any finite product space)
def cmat(ctx, names, states):
    idx = [names.index(v) for v in ctx]; A = np.zeros((2 ** len(ctx), len(states)))
    for i, s in enumerate(states): A[int("".join(str(s[j]) for j in idx), 2), i] = 1
    return A
def feasible(Mat, rhs, n, Aub=None, bub=None):
    return linprog(np.zeros(n), A_eq=Mat, b_eq=rhs, A_ub=Aub, b_ub=bub, bounds=[(0, None)] * n, method="highs").status == 0
def farkas(Mat, rhs):
    """y with Mat^T y >= 0 and rhs.y = -1 (certificate that {p >= 0: Mat p = rhs} is empty)"""
    m = len(rhs); r = linprog(np.zeros(m), A_ub=-Mat.T, b_ub=np.zeros(Mat.shape[1]), A_eq=rhs[None], b_eq=[-1.0], bounds=[(-1e3, 1e3)] * m, method="highs")
    return None if r.status != 0 else r.x
def wrong_side_system(hc):
    """{p >= 0, s >= 0: M_h p = h, g p - s = tau} as an equality system in (p, s)"""
    Mh = np.vstack([O.ctxmat(c) for c in hc]); return Mh, Mh @ P
def margin_dual(hc, q_ctx, tau):
    Mh = np.vstack([O.ctxmat(c) for c in hc]); mh = Mh @ P; Mw = O.ctxmat(q_ctx); q = Mw @ P; k, nh = Mw.shape[0], Mh.shape[0]
    gp, taup = -g, -tau
    res = linprog(-np.r_[q, mh, -taup], A_ub=np.hstack([Mw.T, Mh.T, -gp[:, None]]), b_ub=np.zeros(NS), bounds=[(-1, 1)] * k + [(None, None)] * nh + [(0, None)], method="highs")
    return -res.fun, res.x[:k], Mw, q

# ------------------------------------------------------------------ A1: ZA1, ZA2
ok1 = True; ok2 = True; n_inf = n_zero = n_pos = 0; dual_records = []
for anc in ANCH:
    hc = h_ctx + anc
    ok1 &= O.extends([(c, O.ctxmat(c) @ P) for c in hc + [MCY]])
    for tau in TAUS:
        Mh, mh = wrong_side_system(hc)
        wrong = feasible(Mh, mh, NS, Aub=-g[None], bub=[-tau])
        for q_ctx in INSP:
            r, p0 = O.rho(hc, [q_ctx], g, tau, P)
            if not wrong:
                Mat = np.vstack([np.hstack([Mh, np.zeros((len(mh), 1))]), np.r_[g, -1.0][None]]); rhs = np.r_[mh, tau]
                y = farkas(Mat, rhs); ok1 &= (not np.isfinite(r)) and y is not None and (Mat.T @ y).min() >= -1e-7; n_inf += 1
            else:
                Mq = O.ctxmat(q_ctx); match = feasible(np.vstack([Mh, Mq]), np.r_[mh, Mq @ P], NS, Aub=-g[None], bub=[-tau])
                ok1 &= (r < 1e-9) == match
                if r < 1e-9: n_zero += 1
                else:
                    n_pos += 1; dual, lam, Mw, q = margin_dual(hc, q_ctx, tau); ok2 &= abs(dual - r) < 1e-7
                    for _ in range(20):
                        _cfg, pw = O.far_side_config(hc, [MCY], g, tau, P, rng)
                        if pw is not None: ok2 &= lam @ (Mw @ pw - q) <= -r + 1e-8
                    dual_records.append(dict(anchors=["".join(c) for c in anc], tau=round(tau - gstar, 3), inspection="".join(q_ctx), rho=float(r), lam=[round(x, 4) for x in lam]))
check("ZA1 IDENTITY local-to-outer LP: honest extension feasible; empty wrong side -> rho = +inf with a Farkas certificate; rho = 0 iff a wrong-side law matches the inspection",
      ok1, f"{len(ANCH) * len(TAUS) * len(INSP)} cases: wrong side empty {n_inf}, rho = 0 {n_zero}, rho > 0 {n_pos}")
# SMALL four-variable cycle world for infeasibility witnesses
names4 = ("W", "X", "Y", "Z"); S4 = list(itertools.product((0, 1), repeat=4)); corr = np.array([0.5, 0, 0, 0.5]); anti = np.array([0, 0.5, 0.5, 0])
local4 = [(("W", "X"), corr), (("Y", "Z"), corr)]
outer4 = {"XY+": (("X", "Y"), corr), "ZW-": (("W", "Z"), anti), "W": (("W",), np.array([0.5, 0.5])), "X": (("X",), np.array([0.5, 0.5])), "WX-": (("W", "X"), anti)}
def sys4(blocks): return np.vstack([cmat(c, names4, S4) for c, _ in blocks]), np.concatenate([t for _, t in blocks])
farkas_records = []
for nm in ("WX-",):
    Mat, rhs = sys4(local4 + [outer4[nm]]); y = farkas(Mat, rhs); ok2 &= y is not None; farkas_records.append(dict(outer=[nm], y=[round(v, 3) for v in y]))
Mat, rhs = sys4(local4 + [outer4["XY+"], outer4["ZW-"]]); y = farkas(Mat, rhs); ok2 &= y is not None; farkas_records.append(dict(outer=["XY+", "ZW-"], y=[round(v, 3) for v in y]))
check("ZA2 IDENTITY positive margins equal their dual values and the dual certificate separates; Farkas certificates for empty outer fibres stored separately",
      ok2, f"{n_pos} positive margins (dual = rho, separation on sampled wrong-side laws); {len(farkas_records)} Farkas certificates on the cycle world")
OUT["protection_witnesses"] = dual_records[:6]; OUT["infeasibility_witnesses"] = farkas_records
# ZA3
def infeasible4(keys): Mat, rhs = sys4(local4 + [outer4[k] for k in keys]); return not feasible(Mat, rhs, 16)
keys = list(outer4); exhaustive = []
for r in range(1, len(keys) + 1):
    for sub in itertools.combinations(keys, r):
        if any(set(m) <= set(sub) for m in exhaustive): continue
        if infeasible4(sub): exhaustive.append(sub)
def deletion_min(items, infeas):
    cur = list(items)
    for it in list(cur):
        trial = [x for x in cur if x != it]
        if infeas(trial): cur = trial
    return cur
ok = True
for order in itertools.permutations(keys):
    dm = deletion_min(order, infeasible4); ok &= tuple(sorted(dm)) in [tuple(sorted(m)) for m in exhaustive]
Ps = W.shifted(); outer26 = [("M", "Y"), ("M", "R", "Y"), ("M", "H", "Y"), ("M", "C", "R", "Y"), ("C", "Y"), ("M",)]
def infeasible26(ctxs): return not O.extends([(c, O.ctxmat(c) @ P) for c in DEV] + [(c, O.ctxmat(c) @ Ps) for c in ctxs])
dm26 = deletion_min(outer26, infeasible26)
ok &= infeasible26(dm26) and all(not infeasible26([x for x in dm26 if x != it]) for it in dm26)
check("ZA3 IDENTITY minimal outer witnesses: exhaustive on the cycle world; deletion-based results are inclusion-minimal and among the exhaustive ones",
      ok, f"cycle world minimal witnesses {[list(m) for m in exhaustive]}; Study 26 (other-population outer tables), inclusion-minimal: {['/'.join(c) for c in dm26]}")
OUT["minimal_witnesses_cycle"] = [list(m) for m in exhaustive]; OUT["inclusion_minimal_s26"] = ["".join(c) for c in dm26]

# ------------------------------------------------------------------ A2: ZA4, ZA5
def attribution(blocks):
    names = [n for n, _, _ in blocks]; idx = range(len(blocks))
    def feas(sub): return O.extends([(c, t) for k, (n, c, t) in enumerate(blocks) if k in sub])
    allfe = feas(set(idx)); conflicts, expl = [], []
    if not allfe:
        for r in range(1, len(blocks) + 1):
            for sub in itertools.combinations(idx, r):
                if any(set(m) <= set(sub) for m in conflicts): continue
                if not feas(set(sub)): conflicts.append(sub)
        for r in range(1, len(blocks) + 1):
            for sub in itertools.combinations(idx, r):
                if any(set(m) <= set(sub) for m in expl): continue
                if feas(set(idx) - set(sub)): expl.append(sub)
    named = set.intersection(*[set(e) for e in expl]) if expl else set()
    lab = lambda S: ["/".join(names[k] for k in s) for s in S]
    return dict(consistent=allfe, conflicts=lab(conflicts), explanations=lab(expl), named=sorted(names[k] for k in named)), feas
def blocks_from(laws):  # laws: dict ctx -> law
    return [("".join(c), c, O.ctxmat(c) @ laws.get(c, P)) for c in DEV]
worlds = {}
worlds["1 one poisoned source"] = (blocks_from({("M", "C"): Ps}), {"MC"})
far = None
h6 = [c for c in DEV if c not in (MCY, RHY)]
_cfg, far = O.far_side_config(h6, [MCY, RHY], g, gstar + 0.04, P, np.random.default_rng(271))
b2 = blocks_from({MCY: far, RHY: far}) + [("MY(anchor)", ("M", "Y"), O.ctxmat(("M", "Y")) @ P)]
worlds["2 coherent pair + anchor"] = (b2, {"MCY", "RHY"})
worlds["2' coherent pair, no anchor"] = (blocks_from({MCY: far, RHY: far}), {"MCY", "RHY"})
worlds["3 honest lag"] = (blocks_from({("M", "C"): Ps, ("M", "R"): Ps, ("M", "H"): Ps}), {"MC", "MR", "MH"})
ok = True; rec = {}
for nm, (blocks, planted) in worlds.items():
    a, feas = attribution(blocks); names = [n for n, _, _ in blocks]
    planted_is_expl = feas({k for k, n in enumerate(names) if n not in planted})
    if planted_is_expl: ok &= set(a["named"]) <= planted
    a.update(planted=sorted(planted), planted_is_explanation=bool(planted_is_expl),
              planted_minimal=any(set(e.split("/")) == planted for e in a["explanations"]),
              every_conflict_meets_planted=all(set(c.split("/")) & planted for c in a["conflicts"]), n_explanations=len(a["explanations"]))
    rec[nm] = a
check("ZA4 IDENTITY the procedure names only planted sources whenever the planted set is an explanation", ok,
      "; ".join(f"{k}: named {v['named']}" for k, v in rec.items()))
check("ZA5 FINDING attribution record per world", True,
      " | ".join(f"{k}: consistent {v['consistent']}, conflicts {v['conflicts']}, minimal explanations {v['explanations']}, planted minimal {v['planted_minimal']}" for k, v in rec.items()))
OUT["attribution"] = rec

# ------------------------------------------------------------------ A3: ZA6
tau = gstar + 0.04; rows = []
for q_ctx in INSP:
    r, _ = O.rho(h_ctx, [q_ctx], g, tau, P); q = O.ctxmat(q_ctx) @ P; eta = 3 * np.sqrt(q * (1 - q) / 3000).sum(); c = len(q)
    R = r / (eta * c) if np.isfinite(r) else np.inf; Nst = 3000 * (2 * eta / r) ** 2 if r > 1e-12 else np.inf
    rows.append(dict(inspection="".join(q_ctx), rho=float(r), eta=float(eta), cells=c, ratio=float(R), eligible=bool(r > 2 * eta), N_needed=float(Nst)))
rows.sort(key=lambda x: -x["ratio"])
check("ZA6 FINDING inspection ranking (descriptive) and eligibility (rho > 2 eta) at 3000 episodes", not next(x for x in rows if x["inspection"] == "MY")["eligible"],
      "; ".join(f"{x['inspection']}: R {x['ratio']:.3f}, eligible {x['eligible']}, N* {x['N_needed']:.0f}" for x in rows))
OUT["ranking"] = rows

# ------------------------------------------------------------------ A4: ZA7-ZA9
def own_hist(h, src): return [k for s, k in h if s == src]
ok = True; info = []
for m in (1, 2):
    relay = tuple(x for _ in range(m) for x in (("1", "o"), ("3", "o"), ("1", "c"), ("3", "c")))
    seq = tuple(x for _ in range(m) for x in (("1", "o"), ("1", "c"), ("3", "o"), ("3", "c")))
    same = sorted(relay) == sorted(seq) and BR.linking(relay) == BR.linking(seq) and all(own_hist(relay, s) == own_hist(seq, s) for s in "13")
    corev = lambda h: (sum(1 for s, _ in h if s == "1"), sum(1 for s, _ in h if s == "3")); same &= corev(relay) == corev(seq)
    mr, ms = BR.mu123(relay), BR.mu123(seq); ok &= same and mr != 0 and ms == 0; info.append(f"m = {m}: mu relay {mr}, sequential {ms}")
check("ZA7 IDENTITY brokerage pair: equal counts, co-revision counts, linking numbers and own histories; mu_123 separates relay from sequential", ok, "; ".join(info))
def mdepth(w, n=2, D=6):
    if not w: return None
    Mg = BK.magnus(w, D); nz = [len(a) for a, c in Mg.items() if a and abs(c) > 1e-9]; return min(nz) if nz else None
inv = BK._inv
summ = json.load(open(os.path.join(HERE, "masking_summary.json")))
def rold(n, L):
    w = []
    for _ in range(L):
        i, j = sorted(rng.choice(np.arange(1, n + 1), 2, replace=False).tolist()); gg = SD.pure_gen(i, j); w += gg if rng.random() < 0.5 else inv(gg)
    return w
ok = True; rows8 = []
for k, w in summ["shortest"].items():
    if int(k) > 4: continue
    for _ in range(3):
        bp = SD.section_far(rold(2, 1), 2) + SD.from_loop_word(w, 2) + SD.section_far(rold(2, 1), 2)
        Dw = SD.loop_word(SD.defect(bp, SD.forget(bp, 3, 3), 2), 2); ok &= mdepth(Dw) == mdepth(w) == int(k)
A_ = lambda i, j: SD.pure_gen(i, j); com = lambda u, v: u + v + inv(u) + inv(v)
olds = {"[A12,A23]": com(A_(1, 2), A_(2, 3)), "[A13,[A12,A23]]": com(A_(1, 3), com(A_(1, 2), A_(2, 3)))}
for nm, w in olds.items():
    L = BK.longitudes(w, 3)
    for j in (1, 2, 3):
        _, prod = MK.conj_decomposition([("new", j, 1), ("old", w), ("new", j, -1)], 3)
        dD, dl = mdepth(prod, 3), mdepth(L[j], 3); ok &= (dD is None and dl is None) or (dD == dl + 1); rows8.append((nm, j, dl, dD))
check("ZA8 IDENTITY masking (Magnus expansion of braidknot): lone newcomer has its word's depth; riding has depth 1 + longitude depth, the longitude visible one layer lower",
      ok, f"(old history, source, longitude depth, defect depth): {rows8}")
ok = True
for _ in range(60):
    n = int(rng.integers(2, 4)); bp = []
    for _ in range(int(rng.integers(2, 6))):
        bp.append(("old", rold(n, 1)) if rng.random() < 0.5 else ("new", int(rng.integers(1, n + 1)), int(rng.choice([1, -1]))))
    word, old = MK.build(bp, n); _, prod = MK.conj_decomposition(bp, n)
    ok &= SD.loop_word(SD.defect(word, SD.forget(word, n + 1, n + 1), n), n) == prod
check("ZA9 IDENTITY mixed interleavings: the defect is the product of conjugated newcomer turns (no event lower bound asserted)", ok, "60 random histories")

# ------------------------------------------------------------------ graph controls: ZA10, ZA11
ang = np.exp(1j * np.pi / 3 * np.arange(6)); E6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
Dv = dict(name="hex_valid", P=ang.copy(), E=E6); Pinv = ang.copy(); Pinv[[1, 2]] = Pinv[[2, 1]]; Di = dict(name="hex_invalid", P=Pinv, E=E6)
def commute(D, a, b, N=1): return RG.braid_key(D, [(a, 1), (b, 1)], N) == RG.braid_key(D, [(b, 1), (a, 1)], N)
ok = True; bad_valid = []; inv_pairs = []
for a, b in itertools.combinations(range(len(E6)), 2):
    ind = RG.independent(Dv, a, b); cv = commute(Dv, a, b)
    if cv != ind: bad_valid.append((E6[a], E6[b]))
    if RG.independent(Di, a, b) and RG.segments_cross(Di, a, b): inv_pairs.append((E6[a], E6[b], commute(Di, a, b)))
ok = not bad_valid and inv_pairs and all(not c for _, _, c in inv_pairs) and not any(RG.segments_cross(Dv, a, b) for a, b in itertools.combinations(range(len(E6)), 2) if RG.independent(Dv, a, b))
check("ZA10 IDENTITY valid hexagon drawing: twists commute iff edges are independent; invalid drawing: the crossing independent pair does not commute",
      ok, f"valid mismatches {bad_valid}; invalid drawing crossing independent pairs (commute?): {inv_pairs}")
fa = {}
for N in (1, 2):
    hom, inj, off = RG.faithful_on_ball(Dv, N, 3); fa[N] = (hom, inj)
check("ZA11 FINDING faithfulness on the ball of words of length <= 3 (valid hexagon with diameter; bounded check, not a proof)", True,
      f"N = 1: homomorphism {fa[1][0]}, injective {fa[1][1]}; N = 2: homomorphism {fa[2][0]}, injective {fa[2][1]}")
OUT["graph"] = dict(edges=E6, valid_positions=[[round(z.real, 3), round(z.imag, 3)] for z in ang], invalid_swap=[1, 2], faithful={str(k): v for k, v in fa.items()}, L=3)

# ------------------------------------------------------------------ clock invariance: ZA12, ZA13
def ev_word(n): return [(E6[int(rng.integers(len(E6)))], int(rng.choice([1, -1]))) for _ in range(n)]
def to_idx(w): return [(E6.index(e), s) for e, s in w]
def seq_activation(w, start=0):
    act = {start: 0}; order = [start]
    for t, (e, s) in enumerate(w, 1):
        i, j = e
        if i in act and j not in act: act[j] = t; order.append(j)
        elif j in act and i not in act: act[i] = t; order.append(i)
    return set(act), order
def linear_extension(w):
    n = len(w); preds = {k: {i for i in range(k) if O.dependent(w[i], w[k])} for k in range(n)}; done = []; left = set(range(n))
    while left:
        ready = [k for k in left if preds[k] <= set(done)]; k = ready[int(rng.integers(len(ready)))]; done.append(k); left.discard(k)
    return [w[k] for k in done]
ok_swap = True; ok_reach = True; ok_order = True; n_order_diff = 0
for _ in range(300):
    w = ev_word(int(rng.integers(6, 13))); a0 = O.activation(w, 0); r0, ord0 = seq_activation(w)
    w2 = list(w)
    for _ in range(20):
        i = int(rng.integers(len(w2) - 1))
        if not O.dependent(w2[i], w2[i + 1]): w2[i], w2[i + 1] = w2[i + 1], w2[i]
    ok_swap &= O.activation(w2, 0) == a0
    for _ in range(5):
        le = linear_extension(w); r1, ord1 = seq_activation(le); ok_reach &= r1 == r0
        if ord1 != ord0: ok_order = False; n_order_diff += 1
# ZA12 as specified also required the same total order of activation under retiming. It failed (the order differed in 474 of 1500
# retimings): two members reached through independent events can be reached in either order. Restated (DEVIATIONS_study27a.md): the
# reached set and the activating event of every member are invariant; the total order of activation is not.
ok_act = True
for _ in range(300):
    w = ev_word(int(rng.integers(6, 13))); ids = list(range(len(w)))
    def act_events(seq):
        act = {0: None}
        for k in seq:
            (i, j), _s = w[k]
            if i in act and j not in act: act[j] = k
            elif j in act and i not in act: act[i] = k
        return act
    base = act_events(ids)
    for _ in range(5):
        n = len(w); preds = {k: {i for i in range(k) if O.dependent(w[i], w[k])} for k in range(n)}; done = []; left = set(range(n))
        while left:
            ready = [k for k in left if preds[k] <= set(done)]; k = ready[int(rng.integers(len(ready)))]; done.append(k); left.discard(k)
        ok_act &= act_events(done) == base
check("ZA12 IDENTITY (restated) activation is a trace invariant: unchanged by independent swaps; under retiming the same reached set and the same activating event for every member",
      ok_swap and ok_reach and ok_act, f"swaps: {ok_swap}; reached set under 1500 retimings: {ok_reach}; activating events under 1500 retimings: {ok_act}; "
      f"as first specified (same total order of activation) it failed: the order differed in {n_order_diff} of 1500")
ok_b = True; changed = 0; sound = True; strict = 0
for _ in range(400):
    w = ev_word(int(rng.integers(4, 10))); pos = int(rng.integers(len(w) + 1)); e = E6[int(rng.integers(len(E6)))]
    wp = w[:pos] + [(e, 1), (e, -1)] + w[pos:]
    ok_b &= RG.raag_normal_form(Dv, to_idx(w)) == RG.raag_normal_form(Dv, to_idx(wp)) and RG.braid_key(Dv, to_idx(w), 2) == RG.braid_key(Dv, to_idx(wp), 2)
    changed += set(O.activation(w, 0)) != set(O.activation(wp, 0))
    red = O.reduce(wp)[1]; Rw, Rr = set(O.activation(wp, 0)), set(O.activation(red, 0)); sound &= Rr <= Rw; strict += Rr < Rw
check("ZA13 IDENTITY inverse-pair padding keeps the braid class but can change who is reached; the braid is sound for reach", ok_b and changed > 0 and sound,
      f"braid class unchanged in 400/400 (normal form and N = 2 braid key); reached set changed in {changed}; reduced-word reach inside history reach in 400/400, strictly smaller in {strict}")
json.dump(OUT, open(os.path.join(HERE, "study27a", "results_study27a.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"{sum(results)}/{len(results)} passed")
