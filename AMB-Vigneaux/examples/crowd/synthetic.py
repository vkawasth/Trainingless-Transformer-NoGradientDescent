"""Planted positives and nulls for the crowd pipeline (paper Section sec:crowd): does each detector fire on what is planted
and stay quiet otherwise?

Simulated answers on the REAL design: the same items, the same three workers per item and the same gold labels (so the
real population differences between triples are kept), with each worker's accuracy taken from the real data. Components:
  DS     one-coin Dawid-Skene: workers independent given the truth (the null for B, C, A)
  +dif   item difficulty: 20% of items are hard; on them every worker's error rate rises (logit + 1.5) -- a second
         latent variable (should fire B; E only by pooling)
  +int   worker x difficulty interaction: on hard items, a third of the workers degrade much more (logit + 3.0) and the
         rest only a little (+0.5) -- curvature (should fire C beyond what +dif alone produces)
  +H(l)  a planted four-way obstruction: in 5 tetrahedral covers, on gold-0 items of their triples, with probability l
         the three answers are drawn uniformly from the even-parity patterns (a higher PR box) (should fire A on those
         covers only)
Detectors are those of explore.py: (A) CF vs sampling floor + Benjamini-Hochberg, raw and gold-0; (B) co-error O/E;
(C) worker x difficulty loop test (Q/df); (D) pivotal errors, DS vs MV; (E) latent classes per triple.
(P) Power of (A) for one planted cover as a function of items per triple n and strength l, with a calibrated floor.
Env: CROWD_DIR.
"""
import os, sys, json, itertools, collections, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
spec = importlib.util.spec_from_file_location("crowd_explore", os.path.join(here, "explore.py"))
EX = importlib.util.module_from_spec(spec); spec.loader.exec_module(EX)
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction

rng = np.random.default_rng(7)
EVEN = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
expit = lambda x: 1 / (1 + np.exp(-x)); logit = lambda p: np.log(p / (1 - p))


def planted_covers(a, gold, per, k=5, MIN=12, top_k=60):
    tri = collections.defaultdict(list)
    for q, v in per.items():
        if gold[q] == 0:
            for c in itertools.combinations(sorted(v), 3):
                tri[c].append(q)
    top = [w for w, _ in collections.Counter(a.worker).most_common(top_k)]
    covers = [q4 for q4 in itertools.combinations(sorted(top), 4) if min(len(tri.get(c, [])) for c in itertools.combinations(q4, 3)) >= MIN]
    pick = [covers[i] for i in rng.choice(len(covers), size=min(k, len(covers)), replace=False)]
    return pick, covers


def simulate(per, gold, acc, dif=False, inter=False, lam=0.0, planted=(), seed=0):
    r = np.random.default_rng(seed); workers = sorted(acc); bad = set(r.choice(workers, size=len(workers) // 3, replace=False))
    hard = {q: (r.random() < 0.2) for q in per}
    out = {}
    for q, v in per.items():
        ans = {}
        for w in v:
            le = logit(1 - acc[w])
            if dif and hard[q]:
                le += (3.0 if w in bad else 0.5) if inter else 1.5
            err = r.random() < expit(le); ans[w] = int(gold[q]) ^ int(err)
        out[q] = ans
    if lam > 0:
        trips = {c for q4 in planted for c in itertools.combinations(q4, 3)}
        for q, v in per.items():
            ws = tuple(sorted(v))
            if gold[q] == 0 and ws in trips and r.random() < lam:
                pat = EVEN[r.integers(4)]; out[q] = dict(zip(ws, pat))
    return out, hard


def oracle_c(gold, per_sim, hard, min_n=8):
    """[C] with the TRUE difficulty (hard / easy) instead of the proxy 'how many others erred'"""
    from amb_vigneaux.lattice_path import fit_region
    cells = collections.defaultdict(lambda: [0, 0])
    for q, v in per_sim.items():
        for w, x in v.items():
            c = cells[(w, "hard" if hard[q] else "easy")]; c[0] += int(x != gold[q]); c[1] += 1
    cells = {k: tuple(v) for k, v in cells.items() if v[1] >= min_n}
    fit = fit_region(cells, sorted({w for w, _ in cells}), {"hard", "easy"})
    return fit["Q"] / fit["df"], fit["p"]


def run_detectors(a, gold, per_sim, planted, hard=None):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A = EX.part_a(a, gold, per_sim, B=100); B = EX.part_b(a, gold, per_sim, B=200); C = EX.part_c(a, gold, per_sim)
        Dd = EX.part_d(gold, per_sim); E = EX.part_e(per_sim, top=4, reps=4)
    P = {tuple(sorted(q)) for q in planted}; PT = {c for q in P for c in itertools.combinations(q, 3)}
    res = {}
    for st in ("all", "gold0"):
        rows = A[st]["rows"]
        hit = [r for r in rows if tuple(r["workers"]) in P]; miss = [r for r in rows if tuple(r["workers"]) not in P]
        share = [r for r in miss if any(c in PT for c in itertools.combinations(r["workers"], 3))]; disj = [r for r in miss if r not in share]
        res[st] = dict(planted_sig=sum(r["q_bh"] < .05 for r in hit), planted_n=len(hit), other_sig=sum(r["q_bh"] < .05 for r in miss), other_n=len(miss),
                       sharing_sig=sum(r["q_bh"] < .05 for r in share), sharing_n=len(share), disjoint_sig=sum(r["q_bh"] < .05 for r in disj), disjoint_n=len(disj))
    return dict(A=res, B_pair0=B["gold0"]["pair_OE"], B_pair0_ci=B["gold0"]["pair_ci"], B_tri0=B["gold0"]["triple_OE"],
                C_p=C["p"], C_Qdf=C["Q"] / C["df"], D_share=Dd["share_of_errors_in_2_1"], D_gain=Dd["ds_acc"] - Dd["mv_acc"],
                E_picks=[e["heldout_pick"] for e in E], C_oracle=(oracle_c(gold, per_sim, hard) if hard is not None else None))


def power(ns=(15, 30, 60, 120, 240), lams=(0.0, 0.25, 0.5, 0.75, 1.0), reps=100, nnull=300):
    """one tetrahedral cover, n items per triple: base answers from a one-coin model (accuracies 0.85-0.95, gold 0);
    planted fraction lam of even-parity patterns; floor = 95% quantile of CF at lam = 0"""
    M4 = ("a", "b", "c", "d"); SC = Scenario({x: (0, 1) for x in M4}, tuple(itertools.combinations(M4, 3)))
    acc = np.array([0.85, 0.9, 0.92, 0.95])
    def draw(n, lam):
        tabs = {}
        for C in SC.contexts:
            idx = [M4.index(x) for x in C]; t = np.zeros(8)
            for _ in range(n):
                if rng.random() < lam:
                    pat = EVEN[rng.integers(4)]
                else:
                    pat = tuple(int(rng.random() > acc[i]) for i in idx)
                t[4 * pat[0] + 2 * pat[1] + pat[2]] += 1
            tabs[C] = (t + .5) / (t + .5).sum()
        return contextual_fraction(EmpiricalModel(SC, tabs)).value
    out = {}
    for n in ns:
        thr = float(np.quantile([draw(n, 0.0) for _ in range(nnull)], 0.95))
        out[n] = {lam: float(np.mean([draw(n, lam) > thr for _ in range(reps)])) for lam in lams}
        print(f"    n = {n:3d} items per triple: floor {thr:.3f}; detection rate at lam " + ", ".join(f"{l:.2f}: {p:.2f}" for l, p in out[n].items()))
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey, green = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae", "#1baf7a"
    V = OUT["variants"]; names = list(V)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(colors=muted, labelsize=7)
    x = np.arange(len(names))
    ax[0].bar(x - 0.27, [V[k]["A"]["gold0"]["planted_sig"] / max(V[k]["A"]["gold0"]["planted_n"], 1) for k in names], width=0.27, color=orange, label="planted covers")
    ax[0].bar(x, [V[k]["A"]["gold0"]["sharing_sig"] / max(V[k]["A"]["gold0"]["sharing_n"], 1) for k in names], width=0.27, color=orange, alpha=0.45, label="sharing a planted triple")
    ax[0].bar(x + 0.27, [V[k]["A"]["gold0"]["disjoint_sig"] / max(V[k]["A"]["gold0"]["disjoint_n"], 1) for k in names], width=0.27, color=grey, label="disjoint covers")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=6.5, rotation=30, ha="right"); ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("share BH-significant (gold = 0)", fontsize=8, color=muted); ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].set_title("(a) [A] four-way obstruction", fontsize=8.5, loc="left", color=ink)
    ax[1].bar(x, [V[k]["B_pair0"] for k in names], color=blue, yerr=[[V[k]["B_pair0"] - V[k]["B_pair0_ci"][0] for k in names], [V[k]["B_pair0_ci"][1] - V[k]["B_pair0"] for k in names]], capsize=2)
    ax[1].axhline(1, color=ink, lw=0.8, ls=":"); ax[1].axhline(OUT["real"]["B_pair0"], color=orange, lw=1, ls="--")
    ax[1].text(len(names) - 0.5, OUT["real"]["B_pair0"] + 0.01, "real data", fontsize=6.5, color=orange, ha="right")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=6.5, rotation=30, ha="right")
    ax[1].set_ylabel("pair co-error O/E (gold = 0)", fontsize=8, color=muted); ax[1].set_title("(b) [B] difficulty", fontsize=8.5, loc="left", color=ink)
    ax[2].bar(x - 0.2, [V[k]["C_Qdf"] for k in names], width=0.4, color=green, label="difficulty proxy (others' errors)")
    ax[2].bar(x + 0.2, [V[k]["C_oracle"][0] for k in names], width=0.4, color=blue, alpha=0.6, label="true difficulty (oracle)")
    ax[2].axhline(1, color=ink, lw=0.8, ls=":"); ax[2].legend(fontsize=6, frameon=False, loc="upper left")
    ax[2].axhline(OUT["real"]["C_Qdf"], color=orange, lw=1, ls="--"); ax[2].text(len(names) - 0.5, OUT["real"]["C_Qdf"] + 0.1, "real data", fontsize=6.5, color=orange, ha="right")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=6.5, rotation=30, ha="right")
    ax[2].set_ylabel("loop residual Q / df", fontsize=8, color=muted); ax[2].set_title("(c) [C] worker × difficulty", fontsize=8.5, loc="left", color=ink)
    P = OUT["power"]
    for n, col in zip(sorted(P, key=int), plt.cm.Blues(np.linspace(0.4, 1, len(P)))):
        L = sorted(P[n], key=float); ax[3].plot([float(l) for l in L], [P[n][l] for l in L], "-o", color=col, ms=3, lw=1.5, label=f"n = {n}")
    ax[3].axhline(0.05, color=muted, lw=0.8, ls=":"); ax[3].set_xlabel("planted strength λ", fontsize=8, color=muted); ax[3].set_ylabel("detection rate", fontsize=8, color=muted)
    ax[3].legend(fontsize=6.5, frameon=False, loc="lower right"); ax[3].set_title("(d) [A] power, one cover", fontsize=8.5, loc="left", color=ink)
    fig.tight_layout(); fig.savefig(os.path.join(here, "crowd_synthetic.pdf")); fig.savefig(os.path.join(here, "crowd_synthetic.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "synthetic_results.json"))))
elif __name__ == "__main__":
    a, gold, per = EX.load()
    cnt = collections.defaultdict(lambda: [0, 0])
    for q, v in per.items():
        for w, x in v.items():
            cnt[w][0] += int(x == gold[q]); cnt[w][1] += 1
    acc = {w: (k + 1) / (n + 2) for w, (k, n) in cnt.items()}
    planted, covers = planted_covers(a, gold, per)
    print(f"real design: {len(per)} items; {len(covers)} gold-0 tetrahedral covers; planting into {len(planted)}")
    real = run_detectors(a, gold, per, planted=())
    variants = {"DS (null)": dict(), "+dif": dict(dif=True), "+dif+int": dict(dif=True, inter=True),
                "+H(0.5)": dict(lam=0.5), "+H(1.0)": dict(lam=1.0), "+dif+int+H(1.0)": dict(dif=True, inter=True, lam=1.0)}
    V = {}
    for k, kw in variants.items():
        sim, hard = simulate(per, gold, acc, planted=planted, seed=1, **kw)
        V[k] = run_detectors(a, gold, sim, planted, hard=hard)
        A0 = V[k]["A"]["gold0"]
        print(f"  {k:16s} [A] gold0 BH-sig planted {A0['planted_sig']}/{A0['planted_n']}, sharing a planted triple {A0['sharing_sig']}/{A0['sharing_n']}, disjoint {A0['disjoint_sig']}/{A0['disjoint_n']}"
              f" | [B] pair O/E {V[k]['B_pair0']:.2f} [{V[k]['B_pair0_ci'][0]:.2f},{V[k]['B_pair0_ci'][1]:.2f}], triple {V[k]['B_tri0']:.2f}"
              f" | [C] Q/df {V[k]['C_Qdf']:.2f} (p {V[k]['C_p']:.1e}), oracle Q/df {V[k]['C_oracle'][0]:.2f} (p {V[k]['C_oracle'][1]:.1e}) | [D] pivotal share {V[k]['D_share']:.2f}, DS-MV {V[k]['D_gain']:+.3f} | [E] K {V[k]['E_picks']}")
    print(f"  {'real data':16s} [B] pair O/E {real['B_pair0']:.2f} | [C] Q/df {real['C_Qdf']:.2f} | [D] pivotal share {real['D_share']:.2f}, DS-MV {real['D_gain']:+.3f}")
    print("(P) power of [A] for one tetrahedral cover")
    Pw = power()
    OUT = dict(planted=[list(q) for q in planted], variants=V, real=real, power={str(n): {str(l): p for l, p in d.items()} for n, d in Pw.items()})
    json.dump(json.loads(json.dumps(OUT, default=lambda o: o.tolist() if hasattr(o, "tolist") else (float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))),
              open(os.path.join(here, "synthetic_results.json"), "w"), indent=1)
    plot(json.load(open(os.path.join(here, "synthetic_results.json"))))
