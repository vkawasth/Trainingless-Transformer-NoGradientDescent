"""Checks for 'The memory at work' (companion note; module pce/dmledger.py). E = 1000 episodes per history. Boundaries calibrated on
1000 honest histories (seeds 20001-21000); evaluation on 200 histories per condition (seeds 1-200), disjoint from calibration.
False-alarm unit: for one history and one family, the event that any relation of that family crosses the boundary at any episode from
N0 = 10 to E (family-wise over relations, sequential over the whole horizon). Rates carry 95% Wilson intervals.
  DL1 IDENTITY  the ledger is the source of truth: walks recomputed from a shuffled ledger are identical, and adding the same constant to
                every report of an episode changes no increment
  DL2 IDENTITY  honest increments have mean zero in every family (the null of the text: exchangeable reports and order given the episode)
  DL3 FINDING   family-wise sequential false-alarm rates on honest evaluation histories: the 95% interval contains 5% for every family,
                plain and discounted
  DL4 FINDING   matched conditions: the pattern of alarms (with intervals) for bias, exact shared news, uneven exposure to news, honest
                cascade, lockstep coalition and staggered coalition, as in the table of the text
  DL5 FINDING   uneven exposure to shared news (honest) produces lean alarms, mostly on B or C, and rarely co-revision alarms: the memory
                detects dependence on the news, not intent; the anchor walks of B and C are both positive in most histories
  DL6 FINDING   anchor follow-up: after a lead-lag alarm naming a leader, a one-sided test of the leader's anchor walk at the horizon says
                'hurts' for most staggered coalitions and almost never for honest cascades; 'helps' appears in a third of cascades and never
                for coalitions; the rest are unclear
  DL7 FINDING   discounting (lambda = 0.99) flags a coalition that starts at episode 600 earlier than the cumulative walk
  DL8 FINDING   priming: an independently supplied correct watch-list flags earlier at the nominal false-alarm rate; a misspecified one
                (S5, S6) misses the order and lead-lag, but its lean walk fires on the bystanders S5 and S6 with the sign 'away from X',
                because the coalition pulls the median; a watch-list chosen from the same history inflates the honest false-alarm rate to
                13-28% (double use of evidence)
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import dmledger as DL
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))
E = 1000; CAL = range(20001, 21001); EV = range(1, 201); F = DL.FAMILIES
pBC = DL.PAIRS.index((1, 2)); lBC = DL.OPAIRS.index((1, 2)); lCB = DL.OPAIRS.index((2, 1))
WATCH = {"tilt": [1, 2], "corev": [pBC], "order": [pBC], "leadlag": [lBC, lCB], "anchor": [1, 2]}
WRONG = {"tilt": [4, 5], "corev": [DL.PAIRS.index((4, 5))], "order": [DL.PAIRS.index((4, 5))], "leadlag": [DL.OPAIRS.index((4, 5)), DL.OPAIRS.index((5, 4))], "anchor": [4, 5]}

# DL1
L = DL.ledger("staggered", 300, 5); inc = DL.increments(L, 300); rng = np.random.default_rng(0)
Ls = [L[i] for i in rng.permutation(len(L))]; inc2 = DL.increments(Ls, 300)
Lc = [(e, p, s, r0 + 0.37 * (e % 3), r1 + 0.37 * (e % 3), (a + 0.37 * (e % 3)) if a is not None else None) for (e, p, s, r0, r1, a) in L]; inc3 = DL.increments(Lc, 300)
same = all(np.array_equal(np.nan_to_num(inc[f], nan=9), np.nan_to_num(inc2[f], nan=9)) for f in F)
shift = all(np.allclose(np.nan_to_num(inc[f], nan=9), np.nan_to_num(inc3[f], nan=9)) for f in F)
check("DL1 IDENTITY walks are recomputed from the ledger alone; a shift shared by every report of an episode changes nothing", same and shift)

# calibration
def cache_walks(scen, seeds, lam=None):
    return [DL.walks(DL.increments(DL.ledger(scen, E, sd), E), lam) for sd in seeds]
CALW = cache_walks("honest", CAL); CALWd = cache_walks("honest", CAL, DL.LAM)
def bound_from(Ws, watch=None):
    out = {}
    for f in F:
        mx = [np.abs(W[f][DL.N0 - 1:][:, watch[f]] if watch else W[f][DL.N0 - 1:]).max() for W in Ws]
        out[f] = float(np.quantile(mx, 1 - DL.ALPHA))
    return out
B = bound_from(CALW); Bd = bound_from(CALWd); Bw = bound_from(CALW, WATCH); Bwrong = bound_from(CALW, WRONG)
B1 = {f: float(np.quantile([np.abs(W[f][DL.N0 - 1:][:, 0]).max() for W in CALW], 1 - DL.ALPHA)) for f in F}   # one fixed relation

# DL2
mean_inc = {f: float(np.nanmean(np.concatenate([np.nanmean(DL.increments(DL.ledger("honest", E, sd), E)[f], axis=0) for sd in range(30001, 30051)]))) for f in F}
check("DL2 IDENTITY honest increments have mean zero in every family", all(abs(v) < 0.01 for v in mean_inc.values()), f"{ {f: round(v, 4) for f, v in mean_inc.items()} }")

# evaluation
EVW = {sc: cache_walks(sc, EV) for sc in DL.SCENARIOS}; EVWd = {sc: cache_walks(sc, EV, DL.LAM) for sc in ("honest", "late")}
def first(W, bounds, watch=None):
    out = {}
    for f in F:
        Z = W[f][DL.N0 - 1:]; idx = watch[f] if watch else list(range(Z.shape[1])); Zs = np.abs(Z[:, idx]); hit = np.argwhere(Zs >= bounds[f])
        if len(hit):
            t = hit[:, 0].min(); j = idx[max((h[1] for h in hit if h[0] == t), key=lambda jj: Zs[t, jj])]
            out[f] = (int(t + DL.N0), int(j), float(np.sign(Z[t, j])))
    return out
A = {sc: [first(W, B) for W in EVW[sc]] for sc in DL.SCENARIOS}
def rate(sc, f, sel=None):
    k = sum(1 for a in A[sc] if f in a and (sel is None or sel(a[f]))); return wilson(k, len(A[sc]))
def med(sc, f):
    v = [a[f][0] for a in A[sc] if f in a]; return float(np.median(v)) if v else None
table = {sc: {f: dict(rate=rate(sc, f), median=med(sc, f)) for f in F} for sc in DL.SCENARIOS}

# DL3
fa = {f: rate("honest", f) for f in F}; fad = {f: wilson(sum(1 for W in EVWd["honest"] if f in first(W, Bd)), len(EV)) for f in F}
check("DL3 FINDING honest family-wise sequential false-alarm rates: every 95% interval contains 5% (plain and discounted)",
      all(lo <= 0.05 <= hi for (_, lo, hi) in list(fa.values()) + list(fad.values())),
      f"plain { {f: tuple(round(x, 3) for x in v) for f, v in fa.items()} }; discounted { {f: tuple(round(x, 3) for x in v) for f, v in fad.items()} }")

# DL4
hi = lambda sc, f: table[sc][f]["rate"][1] >= 0.9                       # lower bound of the interval at least 0.9
lo = lambda sc, f: table[sc][f]["rate"][2] <= 0.12                      # upper bound at most 0.12
pattern = (hi("bias", "tilt") and lo("bias", "corev") and lo("bias", "order") and lo("bias", "leadlag")
           and all(lo("news", f) for f in ("tilt", "corev", "order", "leadlag"))
           and hi("cascade", "order") and hi("cascade", "leadlag") and lo("cascade", "tilt")
           and hi("lockstep", "tilt") and hi("lockstep", "corev") and lo("lockstep", "order") and lo("lockstep", "leadlag")
           and all(hi("staggered", f) for f in ("tilt", "corev", "order", "leadlag")))
check("DL4 FINDING matched conditions give distinct alarm patterns", pattern,
      "; ".join(f"{sc}: " + ", ".join(f"{f} {table[sc][f]['rate'][0]:.2f}" for f in F) for sc in DL.SCENARIOS))

# DL5
anch_pos = []
for W in EVW["exposure"]: anch_pos.append(W["anchor"][-1][1] > 0 and W["anchor"][-1][2] > 0)
check("DL5 FINDING uneven exposure to shared news: honest lean alarms, mostly on B or C; co-revision rare; anchor walks positive",
      hi("exposure", "tilt") and table["exposure"]["corev"]["rate"][0] <= 0.15 and rate("exposure", "tilt", lambda x: x[1] in (1, 2))[0] >= 0.8 and np.mean(anch_pos) >= 0.75,
      f"tilt {table['exposure']['tilt']['rate'][0]:.2f} (on B or C {rate('exposure', 'tilt', lambda x: x[1] in (1, 2))[0]:.2f}); co-revision {table['exposure']['corev']['rate'][0]:.2f}; anchor walks of B and C both positive at the horizon in {np.mean(anch_pos):.2f}")

# DL6
def followup(sc):
    out = []
    for W, a in zip(EVW[sc], A[sc]):
        if "leadlag" not in a: out.append(None); continue
        u, v = DL.OPAIRS[a["leadlag"][1]] if a["leadlag"][2] > 0 else DL.OPAIRS[a["leadlag"][1]][::-1]
        z = W["anchor"][-1][u]; out.append("hurts" if z < -1.645 else ("helps" if z > 1.645 else "unclear"))
    return out
fc, fs = followup("cascade"), followup("staggered")
cnt = lambda xs, k: sum(1 for x in xs if x == k)
check("DL6 FINDING the anchor follow-up: 'leader hurts' marks most coalitions and almost no cascades; 'helps' only cascades",
      cnt(fs, "hurts") / len(fs) >= 0.7 and cnt(fc, "hurts") <= 0.02 * len(fc) and cnt(fs, "helps") == 0 and cnt(fc, "helps") >= 0.2 * len(fc),
      f"cascade: helps {cnt(fc, 'helps')}, unclear {cnt(fc, 'unclear')}, hurts {cnt(fc, 'hurts')}, no alarm {cnt(fc, None)}; staggered: hurts {cnt(fs, 'hurts')}, unclear {cnt(fs, 'unclear')}, helps {cnt(fs, 'helps')}, no alarm {cnt(fs, None)}")

# DL7
md_c = {f: med("late", f) for f in ("tilt", "order", "leadlag")}
md_d = {}
for f in ("tilt", "order", "leadlag"):
    v = [first(W, Bd)[f][0] for W in EVWd["late"] if f in first(W, Bd)]; md_d[f] = float(np.median(v)) if v else None
check("DL7 FINDING discounting flags a coalition starting at episode 600 earlier", all(md_d[f] is not None and md_d[f] < md_c[f] for f in md_c),
      f"median alarm episode, cumulative {md_c}, discounted {md_d}")

# DL8
def primed(sc, bounds, watch):
    return [first(W, bounds, watch) for W in EVW[sc]]
def data_derived(W):
    out = {}
    for f in F:
        j = int(np.argmax(np.abs(W[f][-1])))                             # the relation that looks strongest over the whole history
        Z = np.abs(W[f][DL.N0 - 1:, j]); hit = np.argwhere(Z >= B1[f])
        if len(hit): out[f] = int(hit[0, 0] + DL.N0)
    return out
fa_dd = {f: wilson(sum(1 for W in EVW["honest"] if f in data_derived(W)), len(EV)) for f in F}
fa_w = {f: wilson(sum(1 for a in primed("honest", Bw, WATCH) if f in a), len(EV)) for f in F}
pr = primed("staggered", Bw, WATCH); pw = primed("staggered", Bwrong, WRONG)
bystander = [a["tilt"] for a in pw if "tilt" in a]; away = sum(1 for (_, j, sg) in bystander if j in (4, 5) and sg < 0)
md_p = {f: float(np.median([a[f][0] for a in pr if f in a])) for f in ("tilt", "order", "leadlag")}
miss = {f: wilson(sum(1 for a in pw if f in a), len(EV)) for f in ("tilt", "order", "leadlag")}
check("DL8 FINDING priming: correct and independent prior flags earlier; misspecified misses; data-derived inflates false alarms",
      all(md_p[f] <= med("staggered", f) for f in md_p) and miss["order"][2] <= 0.12 and miss["leadlag"][2] <= 0.12 and away >= 0.9 * len(bystander)
      and min(v[0] for v in fa_dd.values()) >= 0.1 and all(v[1] <= 0.05 <= v[2] or v[0] <= 0.05 for v in fa_w.values()),
      f"staggered median episode plain {[med('staggered', f) for f in md_p]} vs primed {list(md_p.values())}; misspecified watch-list alarm rates { {f: round(v[0], 2) for f, v in miss.items()} } (lean alarms on S5/S6 pointing away from X: {away} of {len(bystander)}); honest false alarms: primed { {f: round(v[0], 3) for f, v in fa_w.items()} }, data-derived { {f: round(v[0], 3) for f, v in fa_dd.items()} }")

# matching statistics (reported, for the text)
def marg(sc):
    I = [DL.increments(DL.ledger(sc, E, sd), E) for sd in range(1, 21)]
    return dict(lean_B=float(np.mean([np.nanmean(x["tilt"][:, 1]) for x in I])), corev_BC=float(np.mean([np.nanmean(x["corev"][:, pBC]) for x in I])),
                order_BC=float(np.mean([np.nanmean(x["order"][:, pBC]) for x in I])), leadlag_BC=float(np.mean([np.nanmean(x["leadlag"][:, lBC]) for x in I])),
                anchor_B=float(np.mean([np.nanmean(x["anchor"][:, 1]) for x in I])))
summ = dict(bounds=B, bounds_discounted=Bd, bounds_primed=Bw, table=table, false_alarm_discounted=fad, followup=dict(cascade=fc, staggered=fs),
            late=dict(cumulative=md_c, discounted=md_d), priming=dict(bystander_away=[away, len(bystander)], primed_false_alarm=fa_w, data_derived_false_alarm=fa_dd, primed_median=md_p, misspecified=miss),
            marginals={sc: marg(sc) for sc in DL.SCENARIOS})
json.dump(summ, open(os.path.join(HERE, "dmledger_summary.json"), "w"), indent=1, default=lambda o: None)
print(f"{sum(results)}/{len(results)} passed")
