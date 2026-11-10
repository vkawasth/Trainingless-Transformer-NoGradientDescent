"""Checks for 'Beliefs under contamination' in the companion note: belief presheaves over admitted sources, conflicts and explanations,
tamper budgets, the five-cycle sign toy, the toy bridge, its matched histories, and the event, braid and shadow layers.
Modules pce/tamper.py, pce/bridgetoy.py, pce/trajknot.py. IDENTITY = an exact instance of a statement proved in the text;
FINDING = a computed fact about the toy, reported as such (exploratory).
  TP1  IDENTITY  exposure: on acyclic context families (join tree; GYO reduces to nothing) every minimal conflict is a pair of overlapping
                 cells; a cell that overlaps nothing is in no conflict; conflicts of three or more cells need a cycle (the five-cycle gives
                 one of size 5); a star of three pair contexts on one variable is acyclic (no cycle exposure), although its cell-intersection
                 graph is a triangle
  TP2  IDENTITY  five-cycle sign toy: the open path with s1 = -1 is coherent; closing the cycle with s5 = +1 gives one conflict, all five
                 cells, whose minimal explanations are the five singletons; recruiting S3 (s3 = -1) restores coherence; keeping S3's
                 earlier report as its own cell turns the recruitment into a conflict of two reports on one context
  TP3  IDENTITY  tamper budget (i): coalition {S3} has budget 0 for A (a point) and [0.365, 1] for B; for 200 laws p of the honest fibre
                 the report M_C p is coherent, the DM's interval lies inside the honest-only interval and contains g(p), and both ends
                 are attained
  TP4  IDENTITY  (ii): the forced residual of a report inside the budget is 0 and of a report outside it is > 0; the Farkas certificate
                 satisfies A^T y >= 0, b.y = -1; its support is reported (it touches all five cells, not only the overlaps)
  TP5  IDENTITY  (iii): with one source poisoned by a random table, the 1-robust interval of B contains the true value and the robust
                 decision is never wrong (accept or investigate) in 30 trials, six per source
  TP6  FINDING   inspection: the chords (X2, X4) and (X3, X5) shrink the worst-case B interval to [0.613, 0.930] (robust accept); the other
                 chords leave it at [0.454, 1]; the durable budget of {S3} for B (all five chords bought) is [0.613, 0.930], above the threshold
  TP7  FINDING   histories: the naive decision is wrong in H3 and H4; the robust decision is right or 'investigate' wherever at most one
                 cell is corrupted; after recruitment (H4, two corrupted) the 1-robust decision rejects wrongly and the false world is
                 the only minimal explanation; keeping S2's earlier report and allowing k = 2 restores 'investigate'
  TP8  FINDING   withholding (H2): rank the missing contexts by the reduction of the robust B width under a declared measure mu_t on the
                 current fibre (the law of the LP solution with a standard Gaussian objective; 12 draws); by the minimum, the mean and the
                 maximum over the draws alike, the withheld context (X3, X4) is first of the six (reduction 1 for every draw) and every other
                 context scores 0; it is the inspection that was refused
  TP9  FINDING   event layer: the state is S = (R, W), reports keyed by (source, context) and the impediment log; operations on different
                 keys commute exactly. Letters carry the payload (agreement of the reported table), so H1 and H3 differ intrinsically.
                 Arrival annotations (snapshot compatible? historically consistent? revision?) are stored apart from the letters: the
                 recruitment in H4 and the correction in H5 both arrive snapshot-compatible, historically inconsistent and revised, so the
                 annotation alone cannot tell them apart; in H4 the contradiction is annotated on the inspection, the letter that arrived
                 second. Order-invariant marks from the full history ('x' in some minimal conflict, '+chg') confine H4's suspects to S2, S3
                 and the inspection. Tier 2 adds the robust decision's changes, with trajectory-relative independence tested on the
                 visited states; the decision letter follows the operations it depends on
  TP10 FINDING   braid layer (declared rules): H1 and H1b, same trace and same final tables, have the same braid; but the replay loop of
                 the honest H1 is non-trivial while that of the recruited H4 is trivial, so the braid class does not flag sabotage here
  TP11 FINDING   shadows: the polygon knot typing is validated on the trefoil, its mirror, the figure-eight and the unknot, and is
                 unchanged under rescaling of an axis; for the six histories no three-coordinate shadow of the closed six-component
                 trajectory is knotted; most shadows are singular (they touch themselves), the rest are unknots
"""
import os, sys, itertools
from collections import Counter
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import tamper as T
from pce import bridgetoy as BT
from pce import trajknot as TK
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261109); n = 5

# TP1
def rand_table(ctx):
    p = rng.dirichlet(np.ones(2 ** len(ctx))); keys = list(itertools.product((1, -1), repeat=len(ctx))); return dict(zip(keys, p))
ok = True; sizes = Counter()
for trial in range(12):
    edges = []; order = list(rng.permutation(n))
    for i in range(1, n): edges.append((int(order[i]), int(order[rng.integers(0, i)])))
    cells = [(f"c{i}", e, rand_table(e)) for i, e in enumerate(edges)] + [("iso", (int(order[0]),), rand_table((0,)))] * 0
    for C in T.conflicts(n, cells): sizes[len(C)] += 1; ok &= len(C) == 2
iso_cells = [("a", (0, 1), rand_table((0, 1))), ("b", (1, 2), rand_table((1, 2))), ("lone", (3, 4), rand_table((3, 4)))]
ok_iso = all("lone" not in C for C in T.conflicts(n, iso_cells))
sign = lambda s: {(1, 1): 0.5 * (s > 0), (-1, -1): 0.5 * (s > 0), (1, -1): 0.5 * (s < 0), (-1, 1): 0.5 * (s < 0)}
cyc = [(f"S{i+1}", (i, (i + 1) % 5), sign(-1 if i == 0 else 1)) for i in range(5)]
big = T.conflicts(n, cyc)
star = [("a", (0, 1), rand_table((0, 1))), ("b", (0, 2), rand_table((0, 2))), ("c", (0, 3), rand_table((0, 3)))]
star_ok = all(len(C) == 2 for C in T.conflicts(n, star)) and not any(T.exposure(star)[1].values())
cyc_exp = all(T.exposure(cyc)[1].values()) and not any(T.exposure(cyc[:4])[1].values())
check("TP1 IDENTITY on acyclic families every minimal conflict is a pair; a non-overlapping cell is in no conflict; the cycle gives size 5; star acyclic",
      ok and ok_iso and [len(C) for C in big] == [5] and star_ok and cyc_exp,
      f"tree conflicts by size {dict(sizes)}; cycle conflicts {big}; star conflicts {T.conflicts(n, star)}, cycle exposure {T.exposure(star)[1]}")

# TP2
path = cyc[:4]; full = cyc; rec = [c if c[0] != "S3" else ("S3", c[1], sign(-1)) for c in cyc]
hist = cyc + [("S3@t2", (2, 3), sign(-1))]
expl = T.explanations(n, full, 1)
check("TP2 IDENTITY five-cycle sign toy: path coherent, cycle not; singleton explanations; recruitment coherent; history gives a pair",
      T.feasible(n, path) and not T.feasible(n, full) and sorted(expl) == [(f"S{i}",) for i in range(1, 6)] and T.feasible(n, rec)
      and ("S3", "S3@t2") in T.conflicts(n, hist),
      f"explanations of size 1: {expl}; conflicts with S3's two reports kept: {T.conflicts(n, hist)}")

# TP3
P, gA, gB, h, Q, S3_in, S3_out, c2, H = BT.histories()
honest = [h[k] for k in ("S1", "S2", "S4", "S5")]
IA, IB = T.interval(n, honest, gA), T.interval(n, honest, gB)
M, b = T.system(n, honest); ok = True; hit_lo = hit_hi = False
for s in range(200):
    p = linprog(rng.normal(size=32), A_eq=M, b_eq=b, bounds=[(0, None)] * 32, method="highs").x
    if s % 3 == 0:
        p2 = linprog(rng.normal(size=32), A_eq=M, b_eq=b, bounds=[(0, None)] * 32, method="highs").x; lam = rng.random(); p = lam * p + (1 - lam) * p2
    rep = ("S3", (2, 3), BT.table(p, (2, 3))); I = T.interval(n, honest + [rep], gB)
    ok &= I is not None and IB[0] - 1e-8 <= I[0] <= gB @ p + 1e-8 and gB @ p - 1e-8 <= I[1] <= IB[1] + 1e-8
    hit_lo |= abs(I[0] - IB[0]) < 1e-7; hit_hi |= abs(I[1] - IB[1]) < 1e-7
check("TP3 IDENTITY tamper budget (i): A budget 0, B budget = honest-only interval; coherent reports stay inside and reach both ends",
      abs(IA[1] - IA[0]) < 1e-9 and abs(IB[0] - 0.3654658786) < 1e-6 and abs(IB[1] - 1) < 1e-9 and ok and hit_lo and hit_hi,
      f"I_H(gA) = [{IA[0]:.4f}, {IA[1]:.4f}], I_H(gB) = [{IB[0]:.4f}, {IB[1]:.4f}]; true gA = gB = {gB @ P:.4f}")

# TP4
fr_in, fr_out = T.forced_residual(n, honest, [S3_in]), T.forced_residual(n, honest, [S3_out])
y, touched = T.farkas(n, honest + [S3_out]); A_, b_ = T.system(n, honest + [S3_out])
valid = y is not None and np.all(A_.T @ y >= -1e-8) and abs(b_ @ y + 1) < 1e-8
check("TP4 IDENTITY (ii): forced residual 0 inside the budget, > 0 outside; a valid Farkas certificate; its support reported",
      fr_in < 1e-9 and fr_out > 0.1 and valid, f"residual at agreement 0.40: {fr_in:.2e}; at 0.10: {fr_out:.3f}; certificate touches {touched}")

# TP5
ok = True; inv = 0; picks = []
for trial in range(30):
    i = trial % 5 + 1; bad = (f"S{i}", BT.CTX[f"S{i}"], rand_table(BT.CTX[f"S{i}"]))
    cells = [h[k] if k != f"S{i}" else bad for k in BT.LABELS]
    d = BT.decide(cells, gB, 1); ok &= d["rob"][0] - 1e-8 <= gB @ P <= d["rob"][1] + 1e-8 and d["robust"] in ("accept", "investigate")
    inv += d["robust"] == "investigate"; picks.append((i, d["robust"]))
check("TP5 IDENTITY (iii): one poisoned source, 1-robust interval contains the truth, decision never wrong", ok, f"30 trials: {Counter(picks)}")

# TP6
out = {}
for ch in [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]:
    d = BT.decide([h[k] for k in BT.LABELS] + [("Q", ch, BT.table(P, ch))], gB, 1); out[ch] = (round(d["rob"][0], 3), round(d["rob"][1], 3), d["robust"])
dur, _ = BT.durable_budget(P, gB)
check("TP6 FINDING inspections (X2,X4) and (X3,X5) make B robustly accepted; the others do not; durable budget above threshold",
      out[(1, 3)][2] == out[(2, 4)][2] == "accept" and all(out[c][2] == "investigate" for c in [(0, 2), (0, 3), (1, 4)]) and dur[0] > 0.5,
      f"{out}; durable budget of {{S3}} for B: [{dur[0]:.3f}, {dur[1]:.3f}]")

# TP7
fin = {k: BT.run(st, gB)[1][-1][2] for k, st in H.items()}
naive_wrong = [k for k in fin if fin[k]["naive"] != "accept"]
robust_ok = all(fin[k]["robust"] in ("accept", "investigate") for k in ("H1", "H1b", "H2", "H3", "H5"))
S2new = ("S2new", (1, 2), c2[2]); cellsR = [h["S1"], h["S2"], S2new, S3_in, h["S4"], h["S5"], Q]
ex1 = [e for e in T.explanations(n, cellsR, 1) if "Q" not in e]
Is = [T.interval(n, [c for c in cellsR if c[0] not in e], gB) for e in T.explanations(n, cellsR, 2) if "Q" not in e]
lo2, hi2 = min(i[0] for i in Is), max(i[1] for i in Is)
check("TP7 FINDING naive wrong in H3, H4; robust right or investigate with one corrupted cell; recruitment beats k = 1; history and k = 2 recover",
      sorted(naive_wrong) == ["H3", "H4"] and robust_ok and fin["H4"]["robust"] == "reject" and ex1 == [("S2",)] and lo2 < 0.5 < hi2,
      f"final (naive, robust): { {k: (fin[k]['naive'], fin[k]['robust']) for k in fin} }; with history, k = 1 explanations {ex1}, k = 2 interval [{lo2:.3f}, {hi2:.3f}]")

# TP8
cur = [h[k] for k in ("S1", "S2", "S4", "S5")]; laws = BT._sample_laws(cur, 12, seed=3); w0 = BT._robust_width(cur, gB); stats = {}
for ctx in BT.CANDIDATES:
    if ctx in {tuple(c[1]) for c in cur}: continue
    r = [w0 - BT._robust_width(cur + [("Qc", ctx, BT.table(p, ctx))], gB) for p in laws]; stats[ctx] = (min(r), float(np.mean(r)), max(r))
ranks = [sorted(stats, key=lambda c: -stats[c][j])[0] for j in range(3)]
check("TP8 FINDING the withheld context is first by min, mean and max reduction under the declared measure; the others score 0",
      ranks == [(2, 3)] * 3 and abs(stats[(2, 3)][0] - 1) < 1e-9 and all(abs(v[2]) < 1e-9 for c, v in stats.items() if c != (2, 3)),
      f"(min, mean, max) reduction of the robust B width: { {str(c): tuple(round(x, 3) for x in v) for c, v in stats.items()} }")

# TP9
typed = {}; tier2 = {}
for k, st in H.items():
    tl, conf = BT.typed_letters(st); src = BT.operation_independence(st); lab2 = {l: t for (l, _), t in zip(st, tl)}
    ind = {(lab2[a], lab2[b]) for a in src for b in src if src[a] != src[b]}; typed[k] = T.trace_normal_form(tl, ind)
    w = BT.two_tier(st, gB); tier2[k] = T.trace_normal_form(w, BT.two_tier_independence(st, gB, w))
marks4 = {x.split(":")[0] for layer in typed["H4"] for x in layer if ":x" in x}
ann = {k: [(l, None if a is None else ("S" if a["snapshot"] else "s") + ("H" if a["history"] else "h") + ("R" if a["revision"] else ""))
           for l, a in BT.annotated(st)] for k, st in H.items()}
letters = {k: [l for l, _ in v] for k, v in ann.items()}
check("TP9 FINDING traces: payload letters separate H1 from H3; recruitment and correction arrive snapshot-compatible, historically inconsistent, revised; final marks confine H4 to S2, S3, Q; tier 2 decisions follow their causes",
      typed["H1"] == typed["H1b"] and letters["H1"] != letters["H3"] and ann["H4"][-1][1] == "ShR" and ann["H5"][-1][1] == "ShR"
      and ann["H4"][5][1] == "sh" and marks4 == {"a2", "a3", "aQ", "c2"} and typed["H4"][-1] == ("c2:x+chg",)
      and tier2["H4"][-1] == ("dB:reject@8",) and tier2["H1"][-1] == ("dB:accept@7",),
      f"letters with arrival annotations (S/s snapshot compatible or not, H/h history consistent or not, R revision): {ann}; final-history marks: {typed}; tier 2: {tier2}")

# TP10
wd = {k: BT.path_word([BT.empty_row(gB)] + BT.run(st, gB)[1])[0] for k, st in H.items()}
inv_ = lambda w: [-x for x in reversed(w)]
same = T.artin_trivial(wd["H1"] + inv_(wd["H1b"]), 5)
loops = {k: T.artin_trivial(BT.loop_word(st, gB), 5) for k, st in H.items()}
check("TP10 FINDING braid layer: H1 and H1b have the same braid; replay loops trivial for {H2, H4}, not for honest H1",
      same and not loops["H1"] and loops["H4"] and loops["H2"], f"replay loop trivial: {loops}")

# TP11
t = np.linspace(0, 2 * np.pi, 61)[:-1]
tre = np.c_[np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)]
f8 = np.c_[(2 + np.cos(2 * t)) * np.cos(3 * t), (2 + np.cos(2 * t)) * np.sin(3 * t), np.sin(4 * t)]
unk = np.c_[np.cos(t), np.sin(t), 0.3 * np.sin(2 * t)]
val_ok = (TK.knot_type(tre)[0] == "left trefoil" and TK.knot_type(tre * [1, 1, -1])[0] == "right trefoil" and TK.knot_type(f8)[0] == "figure-eight"
          and TK.knot_type(unk)[0] == "unknot" and TK.knot_type(tre * [3, 0.2, 1])[0] == "left trefoil")
C = {k: BT.components(st, gB) for k, st in H.items()}
allp = np.vstack(list(C.values())); lo = allp.min(0); span = np.maximum(allp.max(0) - lo, 1e-12); tally = {}
for k, c in C.items():
    z = (c - lo) / span; cnt = Counter()
    for S in itertools.combinations(range(6), 3):
        Pp = TK.clean(z[:, S])
        if len(Pp) < 3: cnt["degenerate"] += 1; continue
        cnt["singular" if TK.singular(z[:, S]) else TK.knot_type(z[:, S])[0]] += 1
    tally[k] = dict(cnt)
knotted = sum(v for d in tally.values() for kk, v in d.items() if kk not in ("singular", "unknot", "degenerate"))
check("TP11 FINDING knot typing validated; no three-coordinate shadow of any history's trajectory is knotted", val_ok and knotted == 0,
      f"shadows (20 per history): {tally}")
print(f"{sum(results)}/{len(results)} passed")
