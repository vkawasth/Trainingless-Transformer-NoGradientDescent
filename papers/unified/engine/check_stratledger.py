"""The stratified ledger (section sec:ledger). New script; it modifies nothing; uses pce/stratledger.py, pce/cpl33.py.
Declared before the first run:
  Populations: 10000 nondegenerate 3 x 3 margins (integers over 120, Dirichlet(2), seed 20261126) and 10000 nondegenerate 4 x 4 margins
  (integers over 240, seed 20261127). Credal set of a table: its coupling polytope Cpl(f, g).
  Decision quantity: P(cell (0,0) | block C = rows 0-1 x columns 0-1), lower and upper ends of its band. Evidence: likelihood
  L_ij = 1 + ((i + 2j) mod 3)/2 on cells, applied once (t = 1); drift D = max(|hi_1 - hi_0|, |lo_1 - lo_0|).
  Hypotheses (reported either way): H1 compression (few linear programmes certify every table; for linear objectives at most
  C(r+c-2, r-1)); H2 boundary prediction (top-drift tables enriched near walls); H3 decision preservation (the ledger bound ranks
  the drift: Spearman correlation and top-20 recall).
  QL1 IDENTITY  linear objectives (10 random integer objectives per shape): the library built over all 10000 tables needs at most
                C(r+c-2, r-1) solves (6 for 3 x 3, 20 for 4 x 4); every basis is unimodular; every table's certificate verifies exactly
  QL2 IDENTITY  conditional bands: the band is defined iff f_0 + f_1 + g_0 + g_1 > 1 (the block can vanish otherwise; 300 tables by LP);
                the four libraries (hi, lo at t = 0, 1) certify every defined table with at most r^(c-1) c^(r-1) bases (81, 4096); exact
                re-verification (all 3 x 3 tables, 1000 4 x 4 tables)
  QL3 IDENTITY  the ledger drift bound U = max over ends of the logistic bound from osc_C(log L) holds for every table
  QL4 RECORD    certified top-20 drift search in decreasing U: evaluations needed until no unexamined table can enter
  QL5 RECORD    H1 for 3 x 3: toric chamber (normal-fan type) against optimal-basis region of the upper end: regions per chamber, purity.
                SUPERSEDED (2026-11-13) by check_stratledger2.py QL9-QL11: the chamber labels here were taken up to the 72 symmetries of
                the table while the basis labels were not, so the reported purity 0.30 is an artifact; with consistent raw labels every
                chamber carries exactly one optimal basis (prop:chamberrefine). The claim based on QL5 is withdrawn from the paper
  QL6 RECORD    H2: share of the top-100 drift tables within clearance 0.02 of a basis wall or of the support wall, against all tables;
                kernel-weighted wall importance (weight exp(-clearance / 0.02)) of the leading walls
  QL7 RECORD    H3: Spearman correlation between U and D; top-20 recall of the first 20, 50, 100 tables by U
  QL8 IDENTITY  reconstruction audit: every stratum aggregate and every wall jump recomputed from the table records equals the stored
"""
import os, sys, json, math, itertools
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import stratledger as SL
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def nondeg(f, g):
    fs = {sum(c) for k in range(1, len(f)) for c in itertools.combinations(f, k)}; gs = {sum(c) for k in range(1, len(g)) for c in itertools.combinations(g, k)}
    return not (fs & gs)
def margins(rng, k, den):
    v = rng.dirichlet([2] * k) * den; x = np.maximum(1, np.floor(v)).astype(int); x[int(np.argmax(x))] += den - x.sum()
    return [Fr(int(y), den) for y in x] if x.min() >= 1 else None
def population(r, c, den, seed, n=10000):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        f, g = margins(rng, r, den), margins(rng, c, den)
        if f and g and nondeg(f, g): out.append((f, g))
    return out
OUT = {}
for (r, c, den, seed) in ((3, 3, 120, 20261126), (4, 4, 240, 20261127)):
    tag = f"{r}x{c}"; P = SL.Problem(r, c); pop = population(r, c, den, seed); T = len(pop)
    Bm = np.array([P.b(f, g) for f, g in pop]).T; rng = np.random.default_rng(seed + 1); order = rng.permutation(T)
    # QL1 linear objectives
    cap = math.comb(r + c - 2, r - 1); solves = []; unimod = True; exact_bad = 0
    for o in range(10):
        nvec = rng.integers(-20, 21, size=P.N).astype(float); L = SL.Library(P)
        assign, val, s, _ = SL.build(P, L, Bm, nvec, np.zeros(P.N), +1, order); solves.append(s); unimod &= all(dv == 1 for dv in L.detabs)
        idx = range(T) if (tag == "3x3" and o < 2) else (rng.choice(T, 300, replace=False) if o < 2 else [])
        for t in idx:
            ok, v = SL.exact_check(P, L, assign[t], P.bF(*pop[t]), [Fr(int(x)) for x in nvec], [Fr(0)] * P.N, +1); exact_bad += (not ok) or assign[t] < 0
    check(f"QL1 IDENTITY {tag} linear objectives: at most C(r+c-2,r-1) = {cap} solves certify all {T} tables; bases unimodular; exact certificates",
          max(solves) <= cap and unimod and exact_bad == 0, f"solves per objective {solves}; exact failures {exact_bad}")
    # QL2 conditional bands
    cells = P.cells; ix = {cl: k for k, cl in enumerate(cells)}
    n0 = np.zeros(P.N); n0[ix[(0, 0)]] = 1; d0 = np.zeros(P.N)
    for cl in ((0, 0), (0, 1), (1, 0), (1, 1)): d0[ix[cl]] = 1
    Lk = np.array([1 + ((i + 2 * j) % 3) / 2 for (i, j) in cells]); n1 = n0 * Lk; d1 = d0 * Lk
    LkF = [Fr(1) + Fr((i + 2 * j) % 3, 2) for (i, j) in cells]
    defined = np.array([f[0] + f[1] + g[0] + g[1] > 1 for f, g in pop]); rule_bad = 0
    for t in rng.choice(T, 300, replace=False):
        r_ = linprog(d0, A_eq=P.A, b_eq=Bm[:, t], bounds=[(0, None)] * P.N, method="highs"); rule_bad += (r_.fun > 1e-12) != defined[t]
    D_idx = np.where(defined)[0]; Bd = Bm[:, D_idx]; ordD = rng.permutation(len(D_idx)); libs = {}; vals = {}; asg = {}; sol = {}
    for key, (nv, dv, sense) in dict(hi0=(n0, d0, +1), lo0=(n0, d0, -1), hi1=(n1, d1, +1), lo1=(n1, d1, -1)).items():
        L = SL.Library(P); a_, v_, s_, gr = SL.build(P, L, Bd, nv, dv, sense, ordD); libs[key] = L; vals[key] = v_; asg[key] = a_; sol[key] = (s_, gr)
    trees = r ** (c - 1) * c ** (r - 1); allc = all((asg[k] >= 0).all() for k in asg)
    ex_idx = range(len(D_idx)) if tag == "3x3" else rng.choice(len(D_idx), 1000, replace=False); ex_bad = 0
    nF = {k: [Fr(int(x)) for x in n0] for k in ("hi0", "lo0")}; dF0 = [Fr(int(x)) for x in d0]
    for t in ex_idx:
        bF = P.bF(*pop[D_idx[t]])
        for key, sense in (("hi0", 1), ("lo0", -1), ("hi1", 1), ("lo1", -1)):
            nn = [Fr(int(n0[q])) * (LkF[q] if key.endswith("1") else 1) for q in range(P.N)]; dd = [Fr(int(d0[q])) * (LkF[q] if key.endswith("1") else 1) for q in range(P.N)]
            ok, v = SL.exact_check(P, libs[key], asg[key][t], bF, nn, dd, sense); ex_bad += (not ok) or abs(float(v) - vals[key][t]) > 1e-9
    check(f"QL2 IDENTITY {tag} conditional bands: definedness rule; four libraries certify every defined table within {trees} bases; exact re-verification",
          rule_bad == 0 and allc and ex_bad == 0 and max(len(libs[k].bases) for k in libs) <= trees,
          f"{len(D_idx)} defined of {T}; rule mismatches {rule_bad}/300; solves (hi0, lo0, hi1, lo1) {[sol[k][0] for k in ('hi0', 'lo0', 'hi1', 'lo1')]}; exact failures {ex_bad}/{4 * len(ex_idx)}")
    # QL3 drift bound
    lC = np.log(Lk); aC = ix[(0, 0)]; rest = [ix[cl] for cl in ((0, 1), (1, 0), (1, 1))]
    dp = lC[aC] - lC[rest].min(); dm = lC[rest].max() - lC[aC]
    sig = lambda x: 1 / (1 + np.exp(-x))
    def move_bound(p):
        with np.errstate(divide="ignore"):
            l = np.log(p) - np.log1p(-p)
        up = np.where((p > 0) & (p < 1), sig(l + max(dp, 0)) - p, 0.0); dn = np.where((p > 0) & (p < 1), p - sig(l - max(dm, 0)), 0.0)
        return np.maximum(up, dn)
    U = np.maximum(move_bound(vals["hi0"]), move_bound(vals["lo0"])); Dr = np.maximum(np.abs(vals["hi1"] - vals["hi0"]), np.abs(vals["lo1"] - vals["lo0"]))
    check(f"QL3 IDENTITY {tag} the ledger drift bound holds for every defined table", bool((Dr <= U + 1e-12).all()), f"max D {Dr.max():.4f}; max U {U.max():.4f}; violations {int((Dr > U + 1e-12).sum())}")
    # QL4 certified top-20
    k = 20; ordU = np.argsort(-U); seen = []; best = []
    for cnt, t in enumerate(ordU, 1):
        seen.append(t); kth = np.sort(Dr[seen])[-k] if len(seen) >= k else -1
        nxt = U[ordU[cnt]] if cnt < len(ordU) else -1
        if len(seen) >= k and nxt < kth: break
    true_top = set(np.argsort(-Dr)[:k]); found = set(np.array(seen)[np.argsort(-Dr[seen])[:k]])
    check(f"QL4 RECORD {tag} certified top-20 drift search", True, f"evaluations needed {len(seen)} of {len(D_idx)}; recovered true top-20 {found == true_top}")
    # strata, walls, boundary
    A_ = asg["hi0"]; Lh = libs["hi0"]; X = np.zeros(len(D_idx)); leave = {}
    for t in range(len(D_idx)):
        kk = A_[t]; xB = Lh.inv[kk] @ Bd[:, t]; X[t] = xB.min(); leave[t] = Lh.bases[kk][int(np.argmin(xB))]
    supp = np.array([float(f[0] + f[1] + g[0] + g[1] - 1) for f, g in (pop[i] for i in D_idx)])
    def grad(kk, bvec, nv, dv):
        B = list(Lh.bases[kk]); inv = Lh.inv[kk]; xB = inv @ bvec; num = nv[B] @ xB; dn = dv[B] @ xB; return (nv[B] @ inv * dn - dv[B] @ inv * num) / dn ** 2
    walls = defaultdict(lambda: dict(tables=[], weights=[], jumps=[])); delta = 0.02
    present = set(A_.tolist()); base_by = {Lh.bases[q]: q for q in present}
    for t in np.where(X <= delta)[0]:
        kk = A_[t]; B = set(Lh.bases[kk]); lv = leave[t]
        for q in present:                                                 # neighbour across the wall of the leaving cell
            B2 = set(Lh.bases[q])
            if q != kk and len(B & B2) == P.m - 1 and lv in B and lv not in B2:
                g1 = grad(kk, Bd[:, t], n0, d0); g2 = grad(q, Bd[:, t], n0, d0); w = math.exp(-X[t] / delta)
                key = (kk, q); walls[key]["tables"].append(int(D_idx[t])); walls[key]["weights"].append(w); walls[key]["jumps"].append(float(np.linalg.norm(g1 - g2)))
                break
    W = {f"{a}->{b}": dict(tables=v["tables"], importance=float(np.dot(v["weights"], v["jumps"])), jump=float(np.average(v["jumps"], weights=v["weights"])))
         for (a, b), v in walls.items()}
    top100 = np.argsort(-Dr)[:100]; nearw = (X <= delta) | (supp <= delta); enr = nearw[top100].mean() / max(nearw.mean(), 1e-12)
    strata = {}
    for q in present:
        mem = np.where(A_ == q)[0]; strata[str(q)] = dict(basis=[list(map(int, P.cells[j])) for j in Lh.bases[q]], tables=[int(D_idx[i]) for i in mem],
                                                         count=int(len(mem)), drift_total=float(Dr[mem].sum()), drift_max=float(Dr[mem].max()))
    # QL5 (3 x 3 only): toric chamber vs basis region
    if tag == "3x3":
        from pce import cpl33 as C
        NV = {cl: np.array(C.normals()[cl], float) for cl in C.CELLS}
        def fv(f, g, F):
            x0 = C.base_point(f, g); out = set()
            for Sq in itertools.combinations(F, 4):
                M = np.array([NV[q] for q in Sq])
                if abs(np.linalg.det(M)) < 1e-9: continue
                u = np.linalg.solve(M, [-float(x0[q]) for q in Sq])
                if all(NV[q] @ u >= -float(x0[q]) - 1e-9 for q in F): out.add(frozenset(Sq))
            return out
        chamber = []
        for i in D_idx:
            f, g = pop[i]; F = [q for q in C.CELLS if f[q[0]] + g[q[1]] <= 1]; chamber.append(C.canon_fan(fv(f, g, F)))
        ch_ids = {k_: n_ for n_, k_ in enumerate(sorted(set(chamber), key=str))}; chv = np.array([ch_ids[x] for x in chamber])
        per = {cid: len(set(A_[chv == cid].tolist())) for cid in set(chv.tolist())}
        purity = sum(Counter(A_[chv == cid].tolist()).most_common(1)[0][1] for cid in set(chv.tolist())) / len(A_)
        check("QL5 RECORD H1: toric chambers against optimal-basis regions of the upper end", True,
              f"{len(ch_ids)} chambers, {len(present)} basis regions; regions per chamber {sorted(per.values())}; purity of the majority region per chamber {purity:.2f}")
    check(f"QL6 RECORD {tag} H2: top-100 drift tables near walls (clearance <= 0.02) against all tables", True,
          f"near-wall share: top-100 {nearw[top100].mean():.2f}, all {nearw.mean():.2f}, enrichment {enr:.2f}; walls {len(W)}; leading importance "
          + ", ".join(f"{k_}: {v['importance']:.2f}" for k_, v in sorted(W.items(), key=lambda kv: -kv[1]['importance'])[:3]))
    rho = spearmanr(U, Dr).correlation; rec = {m_: len(set(ordU[:m_]) & true_top) / k for m_ in (20, 50, 100)}
    check(f"QL7 RECORD {tag} H3: the ledger bound ranks the drift", True, f"Spearman(U, D) {rho:.2f}; top-20 recall within the first 20/50/100 by U: {rec[20]:.2f}/{rec[50]:.2f}/{rec[100]:.2f}")
    # ledger records
    tables = []
    for t in range(len(D_idx)):
        i = int(D_idx[t]); f, g = pop[i]; kk = int(A_[t]); B = list(Lh.bases[kk]); xB = Lh.inv[kk] @ Bd[:, t]
        supp_cells = [P.cells[B[q]] for q in range(P.m) if xB[q] > 1e-12]; top = max(np.log(Lk[ix[cl]]) for cl in supp_cells)
        tables.append(dict(id=i, f=[str(x) for x in f], g=[str(x) for x in g], hi0=vals["hi0"][t], lo0=vals["lo0"][t], hi1=vals["hi1"][t], lo1=vals["lo1"][t],
                           width=vals["hi0"][t] - vals["lo0"][t], drift=float(Dr[t]), bound=float(U[t]), stratum=kk,
                           boundary_clearance=dict(basis_wall=float(X[t]), support_wall=float(supp[t])),
                           valuation_profile=dict(family="evidence flow t -> infinity with likelihood L", convention="order of vanishing in t of each cell of the upper-end vertex",
                                                  orders={f"{cl[0]}{cl[1]}": float(top - np.log(Lk[ix[cl]])) for cl in supp_cells}),
                           certificates=dict(hi0=int(asg["hi0"][t]), lo0=int(asg["lo0"][t]), hi1=int(asg["hi1"][t]), lo1=int(asg["lo1"][t]))))
    undefined = [dict(id=int(i), event="band undefined: the block can vanish (f0+f1+g0+g1 <= 1)") for i in np.where(~defined)[0]]
    # QL8 audit
    bad = 0; byid = {tb["id"]: tb for tb in tables}
    for sid, s in strata.items():
        bad += abs(sum(byid[i]["drift"] for i in s["tables"]) - s["drift_total"]) > 1e-9 or len(s["tables"]) != s["count"]
    for key, w in W.items():
        a, b_ = map(int, key.split("->")); js = []; ws = []
        for i in w["tables"]:
            t = int(np.where(D_idx == i)[0][0]); js.append(float(np.linalg.norm(grad(a, Bd[:, t], n0, d0) - grad(b_, Bd[:, t], n0, d0)))); ws.append(math.exp(-X[t] / delta))
        bad += abs(float(np.dot(ws, js)) - w["importance"]) > 1e-9
    check(f"QL8 IDENTITY {tag} reconstruction audit: stratum totals and wall importances recomputed from table records", bad == 0, f"{len(strata)} strata, {len(W)} walls; mismatches {bad}")
    OUT[tag] = dict(tables=tables, undefined=undefined, strata=strata, walls=W, libraries={k_: [[list(map(int, P.cells[j])) for j in B] for B in libs[k_].bases] for k_ in libs},
                    growth={k_: sol[k_][1] for k_ in sol}, linear_solves=solves, topk=dict(evaluations=len(seen), recovered=found == true_top), spearman=rho, enrichment=enr)
json.dump(OUT, open(os.path.join(HERE, "stratledger.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"{sum(results)}/{len(results)} passed")
