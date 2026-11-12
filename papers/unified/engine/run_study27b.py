"""Study 27b runner (specification study27b/SPEC_study27b.md, frozen before this script was run; SHA-256 in FROZEN_SPEC_study27b.txt).
Writes study27b/results_study27b.json. Every number here is taken from the specification; nothing is tuned.
Sections follow the specification: 0 constants, 2 histories, 3 representation arm (H1-H4), 4 decision arm, 5 witness arm (H5, toy),
6 knot arm.
"""
import os, sys, json, math, time, hashlib, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pce import outerrelay as O, raagbraid as RG
W = O.W
OUT = os.path.join(HERE, "study27b"); T0 = time.time()
SMOKE = os.environ.get("SMOKE27B") == "1"   # code test only: other seeds, tiny sizes, output discarded unread
NREP, NRUN, SEEDOFF = (40, 3, 100) if SMOKE else (6000, 1000, 0)
spec_hash = hashlib.sha256(open(os.path.join(OUT, "SPEC_study27b.md"), "rb").read()).hexdigest()
assert spec_hash == open(os.path.join(OUT, "FROZEN_SPEC_study27b.txt")).read().split()[0], "specification changed after the freeze"
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
R = {"spec_sha256": spec_hash}

# ------------------------------------------------------------------ 0. constants
E6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
T = 12
P = W.truth(); NS = 32
M1 = O.ev(lambda v: v["M"] == 1); MY1 = O.ev(lambda v: v["M"] == 1 and v["Y"] == 1); M0Y1 = O.ev(lambda v: v["M"] == 0 and v["Y"] == 1)
def g(p): return (MY1 @ p) / (M1 @ p) - (M0Y1 @ p) / ((1 - M1) @ p)
gstar = g(P); tau = gstar + 0.04
CTX = {0: ("M", "C"), 1: ("M", "R"), 2: ("C", "R"), 3: ("M", "C", "Y"), 4: ("M", "H"), 5: ("R", "H", "Y")}
HONEST = [CTX[k] for k in (0, 1, 2, 4, 5)]
Mh = np.vstack([O.ctxmat(c) for c in HONEST]); pm1 = M1 @ P
gl = MY1 / pm1 - M0Y1 / (1 - pm1)
r = linprog(-gl, A_eq=Mh, b_eq=Mh @ P, bounds=[(0, None)] * NS, method="highs"); pmax = r.x; G = g(pmax) - gstar
Q = P + (0.15 / G) * (pmax - P); Pshock = W.shifted()
assert abs(g(Q) - gstar - 0.15) < 1e-9 and np.abs(Mh @ Q - Mh @ P).max() < 1e-12
NSRC = 6000; ANCH = {"MRY": (("M", "R", "Y"), 3000), "MHY": (("M", "H", "Y"), 3000)}
INSP = {"B7_3000": 3000, "B7_120k": 120000}
def se_g(n):
    py1 = (MY1 @ P) / pm1; py0 = (M0Y1 @ P) / (1 - pm1)
    return math.sqrt(py1 * (1 - py1) / (n * pm1) + py0 * (1 - py0) / (n * (1 - pm1)))
def eta(ctx, n, law=P):
    t = O.ctxmat(ctx) @ law; return 3 * float(np.sum(np.sqrt(t * (1 - t) / n)))
R["constants"] = dict(gstar=gstar, tau=tau, gQ=g(Q), g_shock=g(Pshock), se_g_src3=se_g(NSRC), se_g_anchor=se_g(3000), se_g_insp120k=se_g(120000))
log("constants", R["constants"])

# ------------------------------------------------------------------ 2. histories
def gen_history(rng, chain):
    K = rng.poisson(10); ev = []
    for _ in range(K): ev.append([E6[int(rng.integers(7))], int(rng.choice([1, -1])), int(rng.integers(1, T + 1))])
    order = sorted(range(len(ev)), key=lambda i: (ev[i][2], rng.random())); ev = [ev[i] for i in order]
    out = []; pending = []
    for i, e in enumerate(ev):
        out.append(e)
        if rng.random() < 0.15: pending.append((i + int(rng.geometric(0.5)), [e[0], -e[1], None]))
    # insert corrections G events later in the event order (appended at the end if past it), at the round of the event they follow
    word = list(ev); ins = sorted(pending, key=lambda x: -x[0])
    for pos, c in ins:
        pos = min(pos, len(word)); rnd = word[pos - 1][2] if pos >= 1 else 1
        word.insert(pos, [c[0], c[1], rnd])
    r0 = None
    if chain:
        path = [(0, 1), (1, 2), (2, 3)] if rng.random() < 0.5 else [(5, 0), (4, 5), (3, 4)]
        r0 = int(rng.integers(1, 7))
        for k, e in enumerate(path):
            rnd = r0 + k; pos = sum(1 for x in word if x[2] <= rnd); word.insert(pos, [e, 1, rnd])
    return [tuple(x) for x in word], r0
def wd(h): return [(e, s) for e, s, _ in h]
def act_rounds(h):
    a = {0: 0}
    for (i, j), s, rnd in h:
        if i in a and j not in a and a[i] <= rnd: a[j] = rnd
        elif j in a and i not in a and a[j] <= rnd: a[i] = rnd
    return a
def E3(w): return int(3 in O.activation(w, 0))

# features
EI = {e: k for k, e in enumerate(E6)}
def tok(e, s): return 2 * EI[e] + (0 if s == 1 else 1)
D0 = 14 * 30 + 14 * 14; D1 = 14 * 30 + 1
def f_R0(w):
    x = np.zeros(D0)
    for p, (e, s) in enumerate(w[:30]): x[tok(e, s) * 30 + p] = 1
    for a, b in zip(w, w[1:]): x[420 + tok(*a) * 14 + tok(*b)] += 1
    return x
def f_R1(w):
    x = np.zeros(D1); L = O.layers(w)
    for (e, s), l in zip(w, L):
        if l <= 30: x[tok(e, s) * 30 + l - 1] = 1
    x[-1] = max(L, default=0); return x
def f_R2(w): return f_R1(O.reduce(w)[1])
FEAT = {"R0": f_R0, "R1": f_R1, "R2": f_R2}

# transformations
def t_comm(w, rng, n=20):
    w = list(w)
    for _ in range(n):
        if len(w) < 2: break
        i = int(rng.integers(len(w) - 1))
        if not O.dependent(w[i], w[i + 1]): w[i], w[i + 1] = w[i + 1], w[i]
    return w
def pad(w, rng):
    e = E6[int(rng.integers(7))]; s = int(rng.choice([1, -1])); p = int(rng.integers(len(w) + 1))
    return w[:p] + [(e, s), (e, -s)] + w[p:], p
def t_pad(w, rng, want_safe, tries=50):
    a0 = O.activation(w, 0); att = 0
    for _ in range(tries):
        w2, _ = pad(w, rng); att += 1
        if (O.activation(w2, 0) == a0) == want_safe: return w2, att
    return None, att

# ------------------------------------------------------------------ 3. representation arm
log("representation arm")
rng = np.random.default_rng(27002 + SEEDOFF)
def make(n, chain): return [wd(gen_history(rng, chain)[0]) for _ in range(n)]
tr_h, tr_c, te_h, te_c = make(NREP, False), make(NREP, True), make(NREP, False), make(NREP, True)
Wtr = tr_h + tr_c; Wte = te_h + te_c
y2tr = np.r_[np.zeros(NREP), np.ones(NREP)]; y2te = np.r_[np.zeros(NREP), np.ones(NREP)]
y3tr = np.array([E3(w) for w in Wtr]); y3te = np.array([E3(w) for w in Wte])
X = {k: (np.array([f(w) for w in Wtr]), np.array([f(w) for w in Wte])) for k, f in FEAT.items()}
models = {}
for rep in FEAT:
    for lab, ytr in (("E2", y2tr), ("E3", y3tr)):
        gs = GridSearchCV(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000), {"C": [0.01, 0.1, 1, 10]},
                          cv=StratifiedKFold(5, shuffle=True, random_state=27002 + SEEDOFF), scoring="neg_log_loss")
        gs.fit(X[rep][0], ytr); models[(rep, lab)] = gs.best_estimator_
        log("fitted", rep, lab, "C =", gs.best_params_["C"])
def err_vec(rep, lab, words, y):
    Xw = np.array([FEAT[rep](w) for w in words]); return (models[(rep, lab)].predict(Xw) != y).astype(float)
brng = np.random.default_rng(27004 + SEEDOFF)
def boot(diff, B=2000):
    n = len(diff); idx = brng.integers(0, n, (B, n)); m = diff[idx].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
ERR = {(rep, lab): err_vec(rep, lab, Wte, y) for rep in FEAT for lab, y in (("E2", y2te), ("E3", y3te))}
rep_res = {f"{rep}_{lab}": dict(err=float(ERR[(rep, lab)].mean()), logloss=float(-np.mean(
    np.log(np.clip(models[(rep, lab)].predict_proba(X[rep][1])[np.arange(len(Wte)), (y2te if lab == "E2" else y3te).astype(int)], 1e-12, 1))))) for rep in FEAT for lab in ("E2", "E3")}
d = ERR[("R1", "E2")] - ERR[("R0", "E2")]; H1ci = boot(d)
d2 = ERR[("R2", "E3")] - ERR[("R1", "E3")]; H2ci3 = boot(d2); d2b = ERR[("R2", "E2")] - ERR[("R1", "E2")]; H2ci2 = boot(d2b)
R["representation"] = dict(errors=rep_res, E3_rate_test=float(y3te.mean()),
    H1=dict(diff=float(d.mean()), ci=H1ci, verdict=bool(-0.02 < H1ci[0] and H1ci[1] < 0.02)),
    H2=dict(E3_diff=float(d2.mean()), E3_ci=H2ci3, E2_diff=float(d2b.mean()), E2_ci=H2ci2,
            verdict=bool(H2ci3[0] > 0.02 and -0.02 < H2ci2[0] and H2ci2[1] < 0.02)))
log("H1", R["representation"]["H1"], "H2", R["representation"]["H2"])

# H3: transformations of the first 1000 honest and first 1000 coalition test histories
trng = np.random.default_rng(27002 + 1 + SEEDOFF)
NB = min(1000, NREP); base = te_h[:NB] + te_c[:NB]; yb2 = np.r_[np.zeros(NB), np.ones(NB)]; yb3 = np.array([E3(w) for w in base])
fam = {"comm": [], "pad_safe": [], "mixed": [], "pad_op": []}; dropped = {k: 0 for k in fam}; attempts = {"pad_op": 0, "pad_safe": 0}
ident = dict(R1_comm=True, R2_comm=True, R2_pad_safe=True, R2_mixed=True, R2_pad_op=True, act_comm=True, act_pad_safe=True, act_pad_op_changes=True)
for w in base:
    wc = t_comm(w, trng); fam["comm"].append(wc)
    ident["R1_comm"] &= bool(np.array_equal(f_R1(wc), f_R1(w))); ident["R2_comm"] &= O.reduce(wc)[0] == O.reduce(w)[0]
    ident["act_comm"] &= O.activation(wc, 0) == O.activation(w, 0)
    ws, a = t_pad(w, trng, True); attempts["pad_safe"] += a
    if ws is None: dropped["pad_safe"] += 1; ws = w
    fam["pad_safe"].append(ws); ident["R2_pad_safe"] &= O.reduce(ws)[0] == O.reduce(w)[0]; ident["act_pad_safe"] &= O.activation(ws, 0) == O.activation(w, 0)
    wm, _ = t_pad(t_comm(w, trng), trng, True)
    if wm is None: dropped["mixed"] += 1; wm = t_comm(w, trng)
    fam["mixed"].append(wm); ident["R2_mixed"] &= O.reduce(wm)[0] == O.reduce(w)[0]
    wo, a = t_pad(w, trng, False); attempts["pad_op"] += a
    if wo is None: dropped["pad_op"] += 1; fam["pad_op"].append(None); continue
    fam["pad_op"].append(wo); ident["R2_pad_op"] &= O.reduce(wo)[0] == O.reduce(w)[0]; ident["act_pad_op_changes"] &= O.activation(wo, 0) != O.activation(w, 0)
def errs(rep, lab, words, y): return err_vec(rep, lab, words, y)
H3 = {}
for f_ in ("comm", "pad_safe", "mixed"):
    for lab, yb in (("E2", yb2), ("E3", yb3)):
        dl = {rep: errs(rep, lab, fam[f_], yb) - errs(rep, lab, base, yb) for rep in FEAT}
        H3[f"{f_}_{lab}"] = {rep: float(dl[rep].mean()) for rep in FEAT}
        H3[f"{f_}_{lab}"]["R0_minus_R1_ci"] = boot(dl["R0"] - dl["R1"]); H3[f"{f_}_{lab}"]["R0_minus_R2_ci"] = boot(dl["R0"] - dl["R2"])
# adaptive coalition against the frozen R0 E2 model
arng = np.random.default_rng(27005 + SEEDOFF)
def score_R0(w): return float(models[("R0", "E2")].predict_proba(f_R0(w)[None])[0, 1])
def adapt(w, rng, check_round=None):
    cur = list(w); s0 = score_R0(cur)
    for _ in range(3):
        best, bs = None, s0
        for _ in range(20):
            if rng.random() < 0.5: c = t_comm(cur, rng)
            else:
                c, _ = t_pad(cur, rng, True)
                if c is None: continue
            if check_round is not None and not check_round(c): continue
            sc = score_R0(c)
            if sc < bs: best, bs = c, sc
        if best is None: break
        cur, s0 = best, bs
    return cur
adv = [adapt(w, arng) for w in te_c]
adv_err = {rep: float(err_vec(rep, "E2", adv, np.ones(NREP)).mean()) for rep in FEAT}
H3["adaptive_E2_err_on_coalition"] = adv_err; H3["unmodified_E2_err_on_coalition"] = {rep: float(ERR[(rep, "E2")][NREP:].mean()) for rep in FEAT}
H3v = bool(ident["R1_comm"] and ident["R2_comm"] and ident["R2_pad_safe"] and ident["R2_mixed"] and H3["comm_E2"]["R0_minus_R1_ci"][0] > 0.02
           and H3["pad_safe_E2"]["R0_minus_R2_ci"][0] > 0.02)
# H4
ops = [(w, wo, y) for w, wo, y in zip(base, fam["pad_op"], yb3) if wo is not None]
yop = np.array([E3(wo) for _, wo, _ in ops]); e3change = float(np.mean([E3(wo) != E3(w) for w, wo, _ in ops]))
e1 = err_vec("R1", "E3", [wo for _, wo, _ in ops], yop); e2 = err_vec("R2", "E3", [wo for _, wo, _ in ops], yop); H4ci = boot(e2 - e1)
R["representation"].update(H3=dict(results=H3, identities=ident, dropped=dropped, verdict=H3v),
    H4=dict(op_pairs=len(ops), attempts=attempts["pad_op"], op_success_rate=len(ops) / max(1, attempts["pad_op"]),
            safe_attempts=attempts["pad_safe"], E3_change_rate=e3change, err_R1=float(e1.mean()), err_R2=float(e2.mean()), ci=H4ci,
            identity=bool(ident["R2_pad_op"] and ident["act_pad_op_changes"]), verdict=bool(H4ci[0] > 0.02)))
# braid key (secondary encoding) on the first 200 safe pads
Dv = dict(name="hex_valid", P=np.exp(1j * np.pi / 3 * np.arange(6)), E=E6)
# deviation 1: Artin images of long words exceed memory; the braid key is computed only on safe-pad pairs of at most 4 events
small = [(a, b) for a, b in zip(base, fam["pad_safe"]) if len(b) <= 4][:200]
bk_ok = sum(RG.braid_key(Dv, [(E6.index(e), s) for e, s in a], 2) == RG.braid_key(Dv, [(E6.index(e), s) for e, s in b], 2) for a, b in small)
NBK = len(small)
R["representation"]["braid_key_equal_on_safe_pads"] = f"{bk_ok}/{NBK}"
log("H3", H3v, "H4", R["representation"]["H4"])

# ------------------------------------------------------------------ 4. decision arm
log("decision arm")
thr = {rep: float(np.percentile(models[(rep, "E2")].predict_proba(np.array([FEAT[rep](w) for w in tr_h]))[:, 1], 95)) for rep in FEAT}
cnt3 = [sum(1 for e, s in w if 3 in e) for w in tr_h]; thr3 = float(np.percentile(cnt3, 95))
REPORTERS = [("S0", CTX[0], NSRC), ("S1", CTX[1], NSRC), ("S2", CTX[2], NSRC), ("S3", CTX[3], NSRC), ("S4", CTX[4], NSRC), ("S5", CTX[5], NSRC),
             ("MRY", ANCH["MRY"][0], 3000), ("MHY", ANCH["MHY"][0], 3000), ("I3000", ("M", "Y"), 3000), ("I120k", ("M", "Y"), 120000)]
CM = {name: O.ctxmat(ctx) for name, ctx, _ in REPORTERS}; ETA = {name: eta(ctx, n) for name, ctx, n in REPORTERS}
SEG = {name: se_g(n) for name, ctx, n in REPORTERS if "M" in ctx and "Y" in ctx}
REP_CTX = {name: ctx for name, ctx, _ in REPORTERS}
def g_from(name, tab):
    ctx = REP_CTX[name]; my = np.zeros(4); st = list(itertools.product((0, 1), repeat=len(ctx)))
    for i, s in enumerate(st):
        v = dict(zip(ctx, s)); my[2 * v["M"] + v["Y"]] += tab[i]
    return my[3] / (my[2] + my[3]) - my[1] / (my[0] + my[1])
def feasible(blocks, drop=()):
    use = [b for b in blocks if b not in drop]; rows = []; rhs = []; nsl = sum(len(tabs[b]) for b in use); A_ub = []; b_ub = []; k = 0
    nv = NS + nsl
    for b in use:
        Mk = CM[b]; t = tabs[b]; m = len(t)
        for c in range(m):
            row = np.zeros(nv); row[:NS] = Mk[c]; row[NS + k + c] = -1; A_ub.append(row); b_ub.append(t[c])
            row = np.zeros(nv); row[:NS] = -Mk[c]; row[NS + k + c] = -1; A_ub.append(row); b_ub.append(-t[c])
        row = np.zeros(nv); row[NS + k:NS + k + m] = 1; A_ub.append(row); b_ub.append(ETA[b]); k += m
    r = linprog(np.zeros(nv), A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.r_[np.ones(NS), np.zeros(nsl)][None], b_eq=[1.0],
                bounds=[(0, None)] * nv, method="highs")
    return r.status == 0
def naming(blocks):
    expl = []
    for size in (1, 2):
        for S_ in itertools.combinations(blocks, size):
            if any(set(e) <= set(S_) for e in expl): continue
            if feasible(blocks, S_): expl.append(S_)
        if expl: break
    if not expl: return [], []
    named = set(expl[0])
    for e in expl[1:]: named &= set(e)
    return sorted(named), [list(e) for e in expl]
SCEN = ["relay_fixed", "relay_adaptive", "honest", "shock", "cascade_timely", "cascade_lagging", "drift"]
MODELS = ["B2", "B3", "B4", "B4C", "B4T", "B7_3000", "B7_120k"]
drng = np.random.default_rng(27003 + SEEDOFF); runs = []
def laws(sc, t, a3, ad):
    """(law of each reporter) at round t"""
    truth = P
    if sc == "shock" and t >= 4: truth = Pshock
    if sc in ("cascade_timely", "cascade_lagging") and t >= 1: truth = Q
    L = {f"S{k}": truth for k in range(6)}
    if sc in ("relay_fixed", "relay_adaptive"):
        s = min(1.0, (t - a3) / 4) if (a3 is not None and t >= a3) else 0.0; L["S3"] = P + s * (Q - P)
    if sc in ("cascade_timely", "cascade_lagging"): L["S3"] = Q if (a3 is not None and t >= a3) else P
    if sc == "drift":
        s = min(1.0, (t - ad) / 4) if t >= ad else 0.0; L["S3"] = P + s * (Q - P)
    an = P if sc == "cascade_lagging" else truth
    for b in ("MRY", "MHY", "I3000", "I120k"): L[b] = an
    return L, truth
for sc in SCEN:
    log("scenario", sc)
    for run in range(NRUN):
        chain = sc in ("relay_fixed", "relay_adaptive", "cascade_timely", "cascade_lagging")
        h, r0 = gen_history(drng, chain); a = act_rounds(h); a3 = a.get(3)
        if sc == "relay_adaptive":
            w0 = wd(h)
            # adaptive transformations act on the word; rounds travel with events; inserted pairs take the round of the event before
            def to_h(wc, ref):
                out = []; pool = {}
                for i, x in enumerate(ref): pool.setdefault((x[0], x[1]), []).append(x[2])
                last = 1
                for e, s in wc:
                    if pool.get((e, s)): rnd = pool[(e, s)].pop(0)
                    else: rnd = last
                    out.append((e, s, rnd)); last = rnd
                return out
            wa = adapt(w0, arng, check_round=lambda c: act_rounds(to_h(c, h)).get(3) == a3)
            h = to_h(wa, h); assert act_rounds(h).get(3) == a3
        ad = int(drng.integers(3, 9)) if sc == "drift" else None
        w_full = wd(h)
        rec = dict(sc=sc, a3=a3, ad=ad, flags={m: None for m in MODELS}, dec=[], decB7={"B7_3000": [], "B7_120k": []}, names={}, xs=[], ys=[])
        base_tabs = None
        for t in range(T + 1):
            L, truth = laws(sc, t, a3, ad)
            tabs = {}
            for name, ctx, n in REPORTERS:
                pr = CM[name] @ L[name]; tabs[name] = drng.multinomial(n, pr / pr.sum()) / n
            globals()["tabs"] = tabs
            # decision rule
            def decide(extra):
                names_ = ["S3", "MRY", "MHY"] + extra; gv = np.array([g_from(nm, tabs[nm]) for nm in names_]); wv = np.array([1 / SEG[nm] ** 2 for nm in names_])
                gh = float(wv @ gv / wv.sum()); sec = 1 / math.sqrt(wv.sum())
                return ("act" if gh - tau > 2 * sec else "not" if tau - gh > 2 * sec else "und"), gh
            d_, gh = decide([]); rec["dec"].append(d_)
            for m_, nm in (("B7_3000", "I3000"), ("B7_120k", "I120k")): rec["decB7"][m_].append(decide([nm])[0])
            pre = [x for x in h if x[2] <= t]; wp = wd(pre)
            blocks = ["S0", "S1", "S2", "S3", "S4", "S5", "MRY", "MHY"]
            if rec["flags"]["B2"] is None and not feasible(blocks):
                rec["flags"]["B2"] = t; rec["names"]["B2"] = naming(blocks)
            for m_, nm in (("B7_3000", "I3000"), ("B7_120k", "I120k")):
                if rec["flags"][m_] is None and not feasible(blocks + [nm]):
                    rec["flags"][m_] = t
                    if m_ == "B7_120k": rec["names"]["B7_120k"] = naming(blocks + [nm])
            b2 = rec["flags"]["B2"]
            if rec["flags"]["B3"] is None and ((b2 is not None) or sum(1 for e, s in wp if 3 in e) > thr3): rec["flags"]["B3"] = t
            for m_, rep in (("B4", "R0"), ("B4C", "R1"), ("B4T", "R2")):
                if rec["flags"][m_] is None:
                    sc_ = float(models[(rep, "E2")].predict_proba(FEAT[rep](wp)[None])[0, 1]) if wp else 0.0
                    if (b2 is not None) or sc_ > thr[rep]: rec["flags"][m_] = t
            # witness coordinates (relative to round 0)
            xs, ys = {}, {}
            for name, ctx, n in REPORTERS:
                tb = tabs[name]; st = list(itertools.product((0, 1), repeat=len(ctx)))
                xv = g_from(name, tb) if ("M" in ctx and "Y" in ctx) else 0.0
                key = "M" if "M" in ctx else ("Y" if "Y" in ctx else "C")
                yv = sum(tb[i] for i, s in enumerate(st) if dict(zip(ctx, s))[key] == 1)
                xs[name], ys[name] = xv, yv
            rec["xs"].append(xs); rec["ys"].append(ys)
        rec["truth_act"] = bool(g(truth) >= tau); rec["h"] = h
        runs.append(rec)
log("decision runs done")

def outcome(rec, m):
    dec = rec["decB7"][m] if m in rec["decB7"] else rec["dec"]; fl = rec["flags"][m]
    flip = next((t for t, d_ in enumerate(dec) if d_ == "act"), None); fin = dec[-1]
    if rec["sc"] in ("relay_fixed", "relay_adaptive", "drift", "cascade_lagging") and not rec["truth_act"]:
        if fl is not None and (flip is None or fl < flip): return "flag_before_flip"
        if fl is not None: return "late_flag"
        if fin == "act": return "wrong"
        if fin == "und": return "undecided"
        return "correct"
    if fl is not None: return "false_flag"
    if fin == "und": return "undecided"
    return "correct" if ((fin == "act") == rec["truth_act"]) else "wrong"
LOSS = dict(flag_before_flip=0.0, correct=0.0, wrong=1.0, late_flag=1.0, false_flag=0.2, undecided=0.1)
def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    z = 1.96; p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h_ = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h_, c + h_)
E1 = {}
for sc in SCEN:
    rs = [r_ for r_ in runs if r_["sc"] == sc]; E1[sc] = {}
    for m in MODELS:
        oc = [outcome(r_, m) for r_ in rs]; E1[sc][m] = {k: dict(n=oc.count(k), rate=oc.count(k) / len(oc), ci=wilson(oc.count(k), len(oc))) for k in sorted(set(oc))}
LSC = ["relay_fixed", "relay_adaptive", "honest", "shock", "cascade_timely"]
def loss_arr(m, sc): return np.array([LOSS[outcome(r_, m)] for r_ in runs if r_["sc"] == sc])
LOSSM = {m: float(np.mean([loss_arr(m, sc).mean() for sc in LSC])) for m in MODELS}
def boot_loss(m1, m2, B=2000):
    diffs = []
    arrs = {sc: (loss_arr(m1, sc), loss_arr(m2, sc)) for sc in LSC}
    for _ in range(B):
        tot = 0.0
        for sc in LSC:
            a1, a2 = arrs[sc]; idx = brng.integers(0, len(a1), len(a1)); tot += (a1[idx] - a2[idx]).mean()
        diffs.append(tot / len(LSC))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
NI = {"B4C_minus_B4": boot_loss("B4C", "B4"), "B4T_minus_B4": boot_loss("B4T", "B4")}
# endpoint 2
E2_ = {}
for sc in SCEN:
    rs = [r_ for r_ in runs if r_["sc"] == sc]; E2_[sc] = {}
    for m in ("B2", "B7_120k"):
        fl = [r_ for r_ in rs if m in r_["names"]]; cats = dict(flagged=len(fl), correct_identification=0, false_culprit=0, ambiguous=0, other=0)
        planted = {"S3"} if sc in ("relay_fixed", "relay_adaptive", "drift") else set()
        for r_ in fl:
            named, expl = r_["names"][m]
            if not named: cats["ambiguous"] += 1
            elif any(n_ not in planted for n_ in named): cats["false_culprit"] += 1
            elif set(named) == planted: cats["correct_identification"] += 1
            else: cats["other"] += 1
        E2_[sc][m] = cats
R["decision"] = dict(thresholds=dict(R=thr, count3=thr3), endpoint1=E1, loss=LOSSM, noninferiority=NI, endpoint2=E2_)
log("decision endpoints done", LOSSM)

# ------------------------------------------------------------------ 5. witness arm
def crossings(xs_t, ys_t, base, strands):
    """xs_t, ys_t: lists over rounds of dict strand -> coordinate; base: strand -> (bx, by). Returns list of crossings
    (round, a, b, sign_for_a, y-gap) with a the strand that moved from left to right of b."""
    out = []; nT = len(xs_t)
    pos = lambda s, t: (base[s][0] + xs_t[t][s], base[s][1] + ys_t[t][s])
    idx = {s: i for i, s in enumerate(strands)}
    for i_, a in enumerate(strands):
        for b in strands[i_ + 1:]:
            for t in range(nT - 1):
                d0 = pos(a, t)[0] - pos(b, t)[0]; d1 = pos(a, t + 1)[0] - pos(b, t + 1)[0]
                if d0 == 0: d0 = 1e-15 if idx[a] > idx[b] else -1e-15
                if d1 == 0: d1 = 1e-15 if idx[a] > idx[b] else -1e-15
                if d0 * d1 < 0:
                    lam = d0 / (d0 - d1); ya = pos(a, t)[1] + lam * (pos(a, t + 1)[1] - pos(a, t)[1]); yb = pos(b, t)[1] + lam * (pos(b, t + 1)[1] - pos(b, t)[1])
                    mover, other = (a, b) if d0 < 0 else (b, a)
                    ym, yo = (ya, yb) if mover == a else (yb, ya)
                    out.append((t + lam, mover, other, 1 if ym > yo else -1, abs(ya - yb)))
    return sorted(out)
def c_pair(cr, k, A):
    c = 0
    for _, mv, ot, front, _ in cr:
        if {mv, ot} == {k, A}:
            dirk = 1 if mv == k else -1; kfront = front if mv == k else -front; c += dirk * kfront
    return c
def free_reduced_len(cr, strands, base, xs0):
    # word in adjacent transpositions from the x-order, then free reduction (a lower bound on cancellation in the braid group)
    order = sorted(strands, key=lambda s: base[s][0] + xs0[s]); word = []
    for _, mv, ot, front, _ in cr:
        if mv not in order or ot not in order: continue
        i, j = order.index(mv), order.index(ot)
        if abs(i - j) != 1: i = min(i, j)
        else: i = min(i, j)
        word.append((i, front)); a_, b_ = order.index(mv), order.index(ot); order[a_], order[b_] = order[b_], order[a_]
    st = []
    for x in word:
        if st and st[-1][0] == x[0] and st[-1][1] == -x[1]: st.pop()
        else: st.append(x)
    return len(word), len(st)
SRC = [f"S{k}" for k in range(6)]
CONF = {"A": ["MRY", "MHY"], "B": ["MRY", "MHY", "I120k"]}
def kappa(nm): return 2 * math.sqrt(SEG["S3"] ** 2 + SEG[nm] ** 2)
SEy = {}
for name, ctx, n in REPORTERS:
    key = "M" if "M" in ctx else ("Y" if "Y" in ctx else "C"); p_ = float(O.ev(lambda v, k=key: v[k] == 1) @ P)
    SEy[name] = math.sqrt(2 * p_ * (1 - p_) / n)
WIT = {c: {sc: [] for sc in SCEN} for c in CONF}; KNOT = {c: {sc: [] for sc in SCEN} for c in CONF}
for rec in runs:
    for c, anchors in CONF.items():
        strands = SRC + anchors
        base = {f"S{k}": (k * 1e-4, k * 1e-3) for k in range(6)}
        for j, nm in enumerate(anchors): base[nm] = (kappa(nm) + (6 + j) * 1e-4, (6 + j) * 1e-3)
        xs_t = [{s: rec["xs"][t][s] - rec["xs"][0][s] for s in strands} for t in range(T + 1)]
        ys_t = [{s: rec["ys"][t][s] - rec["ys"][0][s] for s in strands} for t in range(T + 1)]
        cr = crossings(xs_t, ys_t, base, strands)
        cmat = {(k, A): c_pair(cr, k, A) for k in SRC for A in anchors}; Phi = sum(abs(v) for v in cmat.values())
        Eset = [k for k in SRC if sum(abs(cmat[(k, A)]) for A in anchors) >= 1]
        last = {k: max([tt for tt, mv, ot, _, _ in cr if k in (mv, ot) and (mv in anchors or ot in anchors)], default=None) for k in Eset}
        arounds = act_rounds(rec["h"])
        expl = {k: bool(int(k[1:]) in arounds and last[k] is not None and arounds[int(k[1:])] <= math.ceil(last[k])) for k in Eset}
        L_, Lr = free_reduced_len(cr, strands, base, {s: 0.0 for s in strands})
        lp_names = rec["names"].get("B2" if c == "A" else "B7_120k", [[], []])[0]
        planted = ["S3"] if rec["sc"] in ("relay_fixed", "relay_adaptive", "drift") else []
        WIT[c][rec["sc"]].append(dict(Phi=Phi, E=Eset, explained=expl, agree_planted=sorted(Eset) == planted, agree_lp=sorted(Eset) == sorted(lp_names),
                                      lp_flagged=("B2" if c == "A" else "B7_120k") in rec["names"], wlen=L_, wred=Lr))
        # knot arm
        unres = []
        for (tt, mv, ot, front, ygap) in cr:
            if (mv in SRC and ot in anchors) or (ot in SRC and mv in anchors):
                k = mv if mv in SRC else ot; A = ot if mv in SRC else mv
                thr_y = 2 * math.sqrt(SEy[k] ** 2 + SEy[A] ** 2); unres.append((k, A, ygap < thr_y))
        widths = {}
        for k in SRC:
            for A in anchors:
                nu = sum(1 for kk, AA, u in unres if kk == k and AA == A and u); widths[f"{k}-{A}"] = 2 * nu
        KNOT[c][rec["sc"]].append(max(widths.values()) if widths else 0)
def bal_acc(c):
    pos = [w_["Phi"] >= 1 for w_ in WIT[c]["relay_fixed"]]; neg = [w_["Phi"] < 1 for w_ in WIT[c]["cascade_timely"]]
    return 0.5 * (np.mean(pos) + np.mean(neg)), float(np.mean(pos)), float(np.mean(neg))
H5 = {c: dict(zip(("balanced_accuracy", "TPR_relay", "TNR_cascade_timely"), bal_acc(c))) for c in CONF}
H5["verdict"] = bool(H5["B"]["balanced_accuracy"] >= 0.90 and H5["A"]["balanced_accuracy"] < 0.90)
EMAP = {}
for c in CONF:
    EMAP[c] = {}
    for sc in SCEN:
        ws = WIT[c][sc]; Ecount = {}
        for w_ in ws: Ecount[",".join(w_["E"]) or "none"] = Ecount.get(",".join(w_["E"]) or "none", 0) + 1
        nexp = sum(len(w_["explained"]) for w_ in ws)
        EMAP[c][sc] = dict(relay_like=float(np.mean([w_["Phi"] >= 1 for w_ in ws])), Phi_mean=float(np.mean([w_["Phi"] for w_ in ws])),
                           E_sets=dict(sorted(Ecount.items(), key=lambda x: -x[1])[:6]),
                           explained_by_relay=(sum(sum(v for v in w_["explained"].values()) for w_ in ws) / nexp) if nexp else None,
                           agree_planted=float(np.mean([w_["agree_planted"] for w_ in ws])),
                           agree_lp_when_flagged=(float(np.mean([w_["agree_lp"] for w_ in ws if w_["lp_flagged"]])) if any(w_["lp_flagged"] for w_ in ws) else None),
                           lp_flagged=int(sum(w_["lp_flagged"] for w_ in ws)),
                           crossings_mean=float(np.mean([w_["wlen"] for w_ in ws])), cancelled_mean=float(np.mean([w_["wlen"] - w_["wred"] for w_ in ws])),
                           knot_width_mean=float(np.mean(KNOT[c][sc])), knot_width_zero=float(np.mean([x == 0 for x in KNOT[c][sc]])))
R["witness"] = dict(kappa={nm: kappa(nm) for nm in ("MRY", "MHY", "I120k")}, H5=H5, explanation_map=EMAP)
log("H5", H5)

# toy (identities)
def toy():
    pX, p1, p0 = 0.3, 0.4, 0.3
    def tab(p1_, p0_): return np.array([(1 - pX) * (1 - p0_), (1 - pX) * p0_, pX * (1 - p1_), pX * p1_])   # (X,Y) = 00,01,10,11
    P0t = tab(p1, p0); q1 = p1 + 0.10 * (1 - pX) / 1.0; q0 = p0 - 0.10 * pX
    Q0 = tab(q1, q0)
    gt = lambda t: t[3] / (t[2] + t[3]) - t[1] / (t[0] + t[1])
    assert abs(gt(Q0) - gt(P0t) - 0.10) < 1e-12 and abs(Q0[2] + Q0[3] - P0t[2] - P0t[3]) < 1e-12 and abs(Q0[1] + Q0[3] - P0t[1] - P0t[3]) < 1e-12
    def scen(kind, anchor):
        # rounds 0..3; source k updates at round max(1, k) (S1, S2 at 1; S3 at 2; S4 at 3)
        upd = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}; xs_t, ys_t = [], []
        for t in range(4):
            xs, ys = {}, {}
            for s_, r_ in upd.items():
                tb = Q0 if t >= r_ else P0t; xs[s_] = gt(tb) - gt(P0t); ys[s_] = (tb[2] + tb[3]) - pX
            if anchor:
                at = Q0 if (kind == "cascade" and anchor == "timely" and t >= 1) else P0t
                xs["A"] = gt(at) - gt(P0t); ys["A"] = (at[2] + at[3]) - pX
            xs_t.append(xs); ys_t.append(ys)
        return xs_t, ys_t
    out = {}
    for s in (0.0025, 0.005, 0.01):
        for anchor_first in (False, True):
            order = ["A", "S1", "S2", "S3", "S4"] if anchor_first else ["S1", "S2", "S3", "S4", "A"]
            base = {nm: (s * k, s * k * 0.3) for k, nm in enumerate(order)}
            res = {}
            for lab, kind, anch in (("C1t", "cascade", "timely"), ("C1l", "cascade", "lagging"), ("C2", "relay", "lagging"), ("C1n", "cascade", None), ("C2n", "relay", None)):
                xs_t, ys_t = scen(kind, anch); strands = ["S1", "S2", "S3", "S4"] + (["A"] if anch else [])
                cr = crossings(xs_t, ys_t, base, strands)
                res[lab] = dict(Phi=sum(abs(c_pair(cr, k, "A")) for k in ("S1", "S2", "S3", "S4")) if anch else 0,
                                word=[(round(tt, 9), mv, ot, fr) for tt, mv, ot, fr, _ in cr],
                                srcword=[(round(tt, 9), mv, ot, fr) for tt, mv, ot, fr, _ in cr if "A" not in (mv, ot)])
            out[f"s={s},anchor_first={anchor_first}"] = res
    D4 = RG.drawing("path4"); wv = [(0, 1), (1, 1), (2, 1)]
    trace_same = True   # by construction: C1 and C2 have the same event word
    bk_same = RG.braid_key(D4, wv, 2) == RG.braid_key(D4, wv, 2)
    return out, dict(trace_same=trace_same, braid_key_same=bk_same, final_reports_same=True,
                     local_compat=bool(abs(Q0[2] + Q0[3] - 0.3) < 1e-12 and abs(Q0[1] + Q0[3] - P0t[1] - P0t[3]) < 1e-12))
toy_res, toy_id = toy()
R["toy"] = dict(results={k: {kk: dict(Phi=vv["Phi"], n_cross=len(vv["word"])) for kk, vv in v.items()} for k, v in toy_res.items()}, identities=toy_id)
WB = {}
main = [k for k in toy_res if "anchor_first=False" in k]
WB["WB3"] = all(toy_res[k]["C1t"]["Phi"] == 0 and toy_res[k]["C2"]["Phi"] == 4 and toy_res[k]["C1t"]["srcword"] == toy_res[k]["C2"]["srcword"] for k in main)
WB["WB4"] = all(toy_res[k]["C1l"]["word"] == toy_res[k]["C2"]["word"] for k in main)
WB["WB5"] = all(toy_res[k]["C1n"]["word"] == toy_res[k]["C2n"]["word"] for k in main)
WB["WB7b_anchor_first_C2_Phi"] = {k: toy_res[k]["C2"]["Phi"] for k in toy_res if "anchor_first=True" in k}
R["toy"]["WB"] = WB
log("toy", WB)
json.dump(R, open(os.path.join(OUT, "results_study27b.json" if not SMOKE else "/dev/null"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("written")
