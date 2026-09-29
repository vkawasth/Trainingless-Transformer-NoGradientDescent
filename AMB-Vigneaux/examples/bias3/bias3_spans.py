"""Span-level bias on BASIL: counts of annotated spans by (source, target camp, polarity), direct aim only.

Level 2: favourability of source s toward camp c:  logit P(pos | s, c).
Level 3: selective bias for sources A, B and camps R, D (the loop A-R-B-D-A):
    Phi = [logit P(pos|A,R) - logit P(pos|A,D)] - [logit P(pos|B,R) - logit P(pos|B,D)]
        = log of the ratio of odds ratios in the 2x2x2 table (source x camp x polarity).
    Phi = 0 is the no-three-way-interaction log-linear model, a toric model. Its toric ideal is generated
    by the single degree-4 binomial of the 2x2x2 table (its Markov basis), and the exact conditional test
    enumerates the one-parameter fiber of tables with all two-way margins fixed.
Robustness: spans cluster within articles, so we also resample EVENTS (cluster bootstrap) for a CI on Phi.
"""
import json, glob, collections, itertools
import numpy as np
from math import lgamma
from event_labels import L, triplet_labels
TL = triplet_labels()
from target_party import party

BASIL = "/tmp/claude-0/BASIL"
WIN = lambda y: "2010-13" if y <= 2013 else ("2014-16" if y <= 2016 else "2017-19")
rng = np.random.default_rng(0)


EXCLUDE = set()


def load_spans():
    rows = []           # (event, source, window, theme, camp_of_target, pol)
    for a in glob.glob(f"{BASIL}/annotations/*/*_ann.json"):
        A = json.load(open(a))
        d = json.load(open(a.replace("/annotations/", "/articles/").replace("_ann.json", ".json")))
        th, _, _ = TL[d["triplet-uuid"]]
        for p in A["phrase-level-annotations"]:
            if p["aim"] != "dir" or p["target"] in EXCLUDE:
                continue
            c = party(p["target"])
            if c in ("R", "D"):
                rows.append((d["triplet-uuid"], d["source"].lower(), WIN(int(d["date"][:4])), th, c, p["polarity"]))
    return rows


def table(rows, A, B, filt=lambda r: True):
    T = np.zeros((2, 2, 2))                  # [source A/B, camp R/D, pol pos/neg]
    for e, s, w, th, c, pol in rows:
        if s in (A, B) and filt((e, s, w, th, c, pol)):
            T[0 if s == A else 1, 0 if c == "R" else 1, 0 if pol == "pos" else 1] += 1
    return T


def phi(T, eps=0.5):
    t = T + eps
    return float(np.log(t[0, 0, 0] * t[0, 1, 1] * t[1, 0, 1] * t[1, 1, 0] / (t[0, 0, 1] * t[0, 1, 0] * t[1, 0, 0] * t[1, 1, 1])))


def exact_no3way(T):
    """Exact conditional test of the no-3-way model on a 2x2x2 table. The fiber (all two-way margins
    fixed) is {T + k*M}, M the Markov-basis move (+1 at even-parity cells, -1 at odd-parity cells).
    P(T) under the model is proportional to 1 / prod n_ijk!."""
    M = np.zeros((2, 2, 2))
    for i, j, k in itertools.product(range(2), repeat=3):
        M[i, j, k] = 1 if (i + j + k) % 2 == 0 else -1
    pos, neg = M > 0, M < 0
    kmin = -int(T[pos].min()); kmax = int(T[neg].min())
    ks = np.arange(kmin, kmax + 1)
    logp = np.array([-sum(lgamma(x + 1) for x in (T + k * M).ravel()) for k in ks])
    p = np.exp(logp - logp.max()); p /= p.sum()
    p_obs = p[list(ks).index(0)]
    return float(p[p <= p_obs * (1 + 1e-9)].sum()), len(ks)


def cluster_boot(rows, A, B, n=4000, filt=lambda r: True):
    ev = sorted({r[0] for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    out = []
    for _ in range(n):
        samp = [r for e in rng.choice(ev, len(ev)) for r in by[e]]
        out.append(phi(table(samp, A, B, filt)))
    return np.percentile(out, [2.5, 97.5])


def fav(T, s):
    t = T[s] + 0.5
    return {c: float(np.log(t[i, 0] / t[i, 1])) for i, c in enumerate(("R", "D"))}


if __name__ == "__main__":
    import sys
    tag = ""
    if "--exclude-trump" in sys.argv:
        EXCLUDE.update({"Donald_Trump", "Trump", "Donald_Trump_Jr."}); tag = "_notrump"
    rows = load_spans()
    print(f"{len(rows)} direct spans at R/D targets")
    res = {}
    for A, B in (("fox", "nyt"), ("fox", "hpo"), ("hpo", "nyt")):
        T = table(rows, A, B)
        p, nf = exact_no3way(T)
        ci = cluster_boot(rows, A, B)
        fa, fb = fav(T, 0), fav(T, 1)
        res[f"{A}-{B}"] = dict(table=T.tolist(), phi=phi(T), exact_p=p, fiber=nf, ci=ci.tolist(), fav={A: fa, B: fb})
        print(f"\n== {A} vs {B}: counts [src][R,D][pos,neg] = {T.astype(int).tolist()}")
        print(f"   favourability log-odds(pos)  {A}: R {fa['R']:+.2f}  D {fa['D']:+.2f}   |   {B}: R {fb['R']:+.2f}  D {fb['D']:+.2f}")
        print(f"   loop Phi = {phi(T):+.2f}   exact no-3-way p = {p:.4g} (fiber size {nf})   event-bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        for w in ("2010-13", "2014-16", "2017-19"):
            f = lambda r, w=w: r[2] == w
            Tw = table(rows, A, B, f); pw, _ = exact_no3way(Tw); ciw = cluster_boot(rows, A, B, n=2000, filt=f)
            res[f"{A}-{B}:{w}"] = dict(table=Tw.tolist(), phi=phi(Tw), exact_p=pw, ci=ciw.tolist())
            print(f"     {w}: Phi = {phi(Tw):+.2f}  exact p = {pw:.4g}  CI [{ciw[0]:+.2f}, {ciw[1]:+.2f}]   n = {int(Tw.sum())}")
        for th in ("ELECT", "INVEST", "POLICY", "SOCIAL"):
            f = lambda r, th=th: r[3] == th
            Tt = table(rows, A, B, f); pt, _ = exact_no3way(Tt)
            res[f"{A}-{B}:{th}"] = dict(table=Tt.tolist(), phi=phi(Tt), exact_p=pt)
            print(f"     {th:6s}: Phi = {phi(Tt):+.2f}  exact p = {pt:.4g}   n = {int(Tt.sum())}")
    json.dump(res, open(f"bias3_spans_results{tag}.json", "w"), indent=1)
