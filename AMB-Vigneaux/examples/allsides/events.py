"""Big events in AllSides coverage of Trump (2015-2020): which layer changes, and how?

Pre-specified events (not scanned): 2016 election, inauguration, Charlottesville, Mueller report, impeachment inquiry,
COVID national emergency. For each event, before = 26 weeks before, after = 26 weeks after (COVID: to data end).
Sources = AllSides sides (left, center, right); topic cells = coarse topic groups with >= 15 articles per side and period.
Clusters for the bootstrap = AllSides story groups (date, topic).

Layer 1  OUTCOME probabilities  P(adversarial article | side), before vs after; gap left - right; inversion = sign flip.
Layer 2  LEAN (natural part)    Delta gap(left - right) pooled over topics.
Layer 3  NATURALITY             Wald chi-square of the non-natural part of Delta b(side, topic): the basic loop changes
                                Delta Phi(s, t) = D(s,t) - D(s,t0) - D(s0,t) + D(s0,t0), covariance by cluster bootstrap.
Time     weekly gap series g(w) = adversarial rate left - right (sentences); Slepian step test (NW = 1, with linear
         term) at the event in a +-52-week window; multitaper band ratios (NW = 3) of 52 weeks before vs after;
         Thomson line F-test after the event.
Bonferroni over the 6 events is applied in the summary.
"""
import sys, json, datetime, collections
import numpy as np
from scipy import stats
sys.path.insert(0, "../..")
import load
from amb_vigneaux.spectral import slepian_step_test, spectral_change, harmonic_ftest

EVENTS = {"election 2016": "2016-11-08", "inauguration": "2017-01-20", "Charlottesville": "2017-08-12",
          "Mueller report": "2019-04-18", "impeachment inquiry": "2019-09-24", "COVID emergency": "2020-03-13"}
SIDES = ("left", "center", "right")
rng = np.random.default_rng(0)
W0 = datetime.date(2015, 6, 15)


def cells(rows, groups, key="outcome"):
    D = np.full((3, len(groups)), np.nan)
    for i, s in enumerate(SIDES):
        for j, g in enumerate(groups):
            x = [r[key] for r in rows if r["side"] == s and r["group"] == g]
            if x:
                D[i, j] = np.mean(x)
    return D


def boot(before, after, groups, B=500):
    def clus(rows):
        c = collections.defaultdict(list)
        for r in rows:
            c[(r["date"], r["topic"])].append(r)
        return list(c.values())
    cb, ca = clus(before), clus(after)
    out = []
    for _ in range(B):
        rb = [r for i in rng.integers(len(cb), size=len(cb)) for r in cb[i]]
        ra = [r for i in rng.integers(len(ca), size=len(ca)) for r in ca[i]]
        out.append((cells(ra, groups) - cells(rb, groups), gap(ra) - gap(rb), gap(rb), gap(ra),
                    gap(ra, "outcome_dir") - gap(rb, "outcome_dir"), gap(rb, "outcome_dir"), gap(ra, "outcome_dir")))
    return out


def gap(rows, key="outcome"):
    L = [r[key] for r in rows if r["side"] == "left"]; R = [r[key] for r in rows if r["side"] == "right"]
    return np.mean(L) - np.mean(R)


def loops(D):
    return np.array([D[s, t] - D[s, 0] - D[0, t] + D[0, 0] for s in range(1, D.shape[0]) for t in range(1, D.shape[1])])


def weekly(R, key_num, key_den):
    n = (max(r["date"] for r in R) - W0).days // 7 + 1
    num = np.zeros((3, n)); den = np.zeros((3, n))
    for r in R:
        if r["date"] < W0:
            continue
        w = (r["date"] - W0).days // 7; i = SIDES.index(r["side"])
        num[i, w] += r[key_num]; den[i, w] += r[key_den]
    rate = num / np.maximum(den, 1)
    return rate, den


if __name__ == "__main__":
    R = load.load(); out = {}
    rate, den = weekly(R, "n_adv", "n_sent")
    g = rate[0] - rate[2]
    ok = (den[0] > 0) & (den[2] > 0)
    g = np.where(ok, g, np.interp(np.arange(len(g)), np.where(ok)[0], g[ok]))
    print(f"weekly series: {len(g)} weeks from {W0}; weeks with both sides present {ok.mean():.2f}")
    for name, ds in EVENTS.items():
        tau = datetime.date.fromisoformat(ds)
        before = [r for r in R if tau - datetime.timedelta(weeks=26) <= r["date"] < tau]
        after = [r for r in R if tau <= r["date"] < tau + datetime.timedelta(weeks=26)]
        cnt = collections.Counter((r["side"], r["group"], r["date"] >= tau) for r in before + after)
        groups = [gname for gname in ("elections", "politics", "foreign", "justice", "immigration", "media", "other", "economy",
                                      "healthcare", "coronavirus")
                  if all(cnt[(s, gname, a)] >= 15 for s in SIDES for a in (False, True))]
        res = dict(n_before=len(before), n_after=len(after), groups=groups)
        # outcome layer
        pb = {s: float(np.mean([r["outcome"] for r in before if r["side"] == s])) for s in SIDES}
        pa = {s: float(np.mean([r["outcome"] for r in after if r["side"] == s])) for s in SIDES}
        bs = boot(before, after, groups)
        gb, ga = gap(before), gap(after)
        se_gb = np.std([b[2] for b in bs]); se_ga = np.std([b[3] for b in bs]); se_d = np.std([b[1] for b in bs])
        res.update(P_before=pb, P_after=pa, gap_before=float(gb), gap_after=float(ga), z_gap_before=float(gb / se_gb),
                   z_gap_after=float(ga / se_ga), delta_gap=float(ga - gb), z_delta_gap=float((ga - gb) / se_d),
                   inversion=bool(np.sign(gb) != np.sign(ga) and abs(gb / se_gb) > 1.96 and abs(ga / se_ga) > 1.96))
        # directed fact-check outcome
        pdb = {s: float(np.mean([r["outcome_dir"] for r in before if r["side"] == s])) for s in SIDES}
        pda = {s: float(np.mean([r["outcome_dir"] for r in after if r["side"] == s])) for s in SIDES}
        db, da = gap(before, "outcome_dir"), gap(after, "outcome_dir"); sdd = np.std([b[4] for b in bs])
        res.update(Pdir_before=pdb, Pdir_after=pda, dir_gap_before=float(db), dir_gap_after=float(da),
                   z_dir_gap_before=float(db / np.std([b[5] for b in bs])), z_dir_gap_after=float(da / np.std([b[6] for b in bs])),
                   z_delta_dir_gap=float((da - db) / sdd))
        # naturality
        if len(groups) >= 2:
            D = cells(after, groups) - cells(before, groups)
            L = loops(D); Lb = np.array([loops(b[0]) for b in bs])
            Lb = Lb[~np.isnan(Lb).any(1)]
            C = np.cov(Lb.T).reshape(len(L), len(L))
            W = float(L @ np.linalg.pinv(C) @ L); df = len(L)
            res.update(naturality_W=W, df=df, p_nonnatural=float(stats.chi2.sf(W, df)))
        # time series
        t = (tau - W0).days // 7
        lo, hi = max(0, t - 52), min(len(g), t + 52)
        st = slepian_step_test(g[lo:hi], t - lo, NW=1)
        m = min(t - lo, hi - t)
        sc = spectral_change(g[t - m:t], g[t:t + m], NW=3) if m >= 20 else None
        hf_a = harmonic_ftest(g[t:t + m], NW=3) if m >= 20 else None
        hf_b = harmonic_ftest(g[t - m:t], NW=3) if m >= 20 else None
        res.update(slepian_step=st, spectral=sc, window_weeks=int(m),
                   line_after=None if hf_a is None else dict(f=hf_a["f_best"], p_adj=hf_a["p_adj"]),
                   line_before=None if hf_b is None else dict(f=hf_b["f_best"], p_adj=hf_b["p_adj"]))
        out[name] = res
        nat = f"W={res['naturality_W']:.1f} df={res['df']} p={res['p_nonnatural']:.3f}" if "naturality_W" in res else "n/a"
        print(f"\n== {name} ({ds}); articles {len(before)} before / {len(after)} after; topic cells {groups}")
        print(f"   outcome P(adversarial) before L/C/R " + "/".join(f"{pb[s]:.3f}" for s in SIDES)
              + "  after " + "/".join(f"{pa[s]:.3f}" for s in SIDES))
        print(f"   gap L-R before {gb:+.3f} (z {gb/se_gb:+.1f})  after {ga:+.3f} (z {ga/se_ga:+.1f})  Delta {ga-gb:+.3f} (z {(ga-gb)/se_d:+.2f})"
              f"  inversion: {res['inversion']}")
        print(f"   DIRECTED P before L/C/R " + "/".join(f"{pdb[s]:.3f}" for s in SIDES) + "  after " + "/".join(f"{pda[s]:.3f}" for s in SIDES)
              + f"  gap L-R {db:+.3f} -> {da:+.3f} (z of change {res['z_delta_dir_gap']:+.2f})")
        print(f"   naturality (non-natural part of Delta b over side x topic): {nat}")
        print(f"   weekly gap: Slepian step {st['step']:+.4f} (z {st['z']:+.2f}, p {st['p']:.3f}, AR rho {st['rho']:.2f})")
        if sc:
            print(f"   multitaper ratios after/before ({m} wk each): " + "  ".join(f"{b['band'][0]}-{b['band'][1]}: {b['ratio']:.2f} (p {b['p']:.3f})" for b in sc))
            print(f"   line F: after f={hf_a['f_best']:.3f} cyc/wk p_adj {hf_a['p_adj']:.3f}; before f={hf_b['f_best']:.3f} p_adj {hf_b['p_adj']:.3f}")
    json.dump(out, open("events_results.json", "w"), indent=1, default=float)
