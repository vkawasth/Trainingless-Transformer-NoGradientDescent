"""Did COVID move the centre toward a toggle?  Outcome = alignment of CENTER outlets in coverage of Trump.

  f = 2 P_C - P_L - P_R      (P_s = share of side s's Trump articles with adversarial / directed fact-check framing)
  f < 0: centre closer to the right; f > 0: closer to the left; the outcome toggles at f = 0.
Windows: 26 weeks before the COVID emergency (2020-03-13) vs the 19 weeks after (to data end).
Placebo: the same before/after statistic at every 2-week step from 2016 to 2020-03 (non-overlapping with COVID after
the cut). Toggle radius before COVID: the change of P_C alone that would flip f (= |f|/2), in units of the placebo sd.
Split of the reading vector: share of f that reads the exact (natural) part of b(side, topic) vs loops (toggle.py).
This is a statement about MEDIA ALIGNMENT, not about votes: no electoral outcome is in these data.
"""
import sys, json, datetime, collections
import numpy as np
sys.path.insert(0, "../..")
import load
from amb_vigneaux.toggle import toggle_radius
from amb_vigneaux.certificate import cells, _design

SIDES = ("left", "center", "right")
COEF = {"left": -1.0, "center": 2.0, "right": -1.0}
rng = np.random.default_rng(0)


def f_of(rows, key):
    P = {s: np.mean([r[key] for r in rows if r["side"] == s]) for s in SIDES}
    return 2 * P["center"] - P["left"] - P["right"], P


def boot_se(rows, key, B=500):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["date"], r["topic"])].append(r)
    ks = list(by); out = []
    for _ in range(B):
        rr = [r for i in rng.integers(len(ks), size=len(ks)) for r in by[ks[i]]]
        out.append(f_of(rr, key)[0])
    return float(np.std(out))


def window(R, a, b):
    return [r for r in R if a <= r["date"] < b]


if __name__ == "__main__":
    R = load.load(); out = {}
    tau = datetime.date(2020, 3, 13); end = max(r["date"] for r in R) + datetime.timedelta(days=1)
    nb, na = datetime.timedelta(weeks=26), end - tau
    for key, name in (("outcome", "adversarial"), ("outcome_dir", "directed")):
        pre, post = window(R, tau - nb, tau), window(R, tau, end)
        f0, P0 = f_of(pre, key); f1, P1 = f_of(post, key)
        s0, s1 = boot_se(pre, key), boot_se(post, key)
        plac = []
        t = datetime.date(2016, 1, 1) + nb
        while t + na <= tau:
            plac.append(f_of(window(R, t, t + na), key)[0] - f_of(window(R, t - nb, t), key)[0]); t += datetime.timedelta(weeks=2)
        plac = np.array(plac); d = f1 - f0
        # reading-vector split on the pre-COVID side x topic table (weights = counts)
        C = cells([r["side"] for r in pre], [r["group"] for r in pre], [r[key] for r in pre], 1)
        edges = sorted(C); S = sorted({e[0] for e in edges}); T = sorted({e[1] for e in edges})
        X = _design(edges, S, T); w = np.array([C[e][2] for e in edges], float)
        ns = {s: sum(C[e][2] for e in edges if e[0] == s) for s in S}
        a = np.array([COEF[e[0]] / ns[e[0]] for e in edges])        # f = sum_e w_e a_e b_e
        b = np.array([C[e][0] for e in edges])
        tr = toggle_radius(a, b, X, w)
        res = dict(P_before=P0, P_after=P1, f_before=float(f0), se_before=s0, f_after=float(f1), se_after=s1, delta=float(d),
                   placebo_sd=float(plac.std()), placebo_pct=float((np.abs(plac) >= abs(d)).mean()),
                   center_move_to_toggle=float(abs(f0) / 2), toggle_in_placebo_sd=float(abs(f0) / plac.std()),
                   share_exact=tr["share_exact"], n_placebo=int(len(plac)))
        out[name] = res
        print(f"== {name}: P before L/C/R " + "/".join(f"{P0[s]:.3f}" for s in SIDES) + "   after " + "/".join(f"{P1[s]:.3f}" for s in SIDES))
        print(f"   f = 2C-L-R: before {f0:+.4f} (se {s0:.4f})  after {f1:+.4f} (se {s1:.4f})  Delta {d:+.4f}")
        print(f"   placebo (n={len(plac)} before/after pairs 2016-2020): sd {plac.std():.4f};  |Delta| >= COVID in {res['placebo_pct']:.2f} of them")
        print(f"   toggle radius before COVID: centre must move {abs(f0)/2:.4f} = {abs(f0)/plac.std():.1f} placebo sd;"
              f"  share of f reading the natural (exact) part {tr['share_exact']:.3f}")
    json.dump(out, open("toggle_covid_results.json", "w"), indent=1, default=float)
