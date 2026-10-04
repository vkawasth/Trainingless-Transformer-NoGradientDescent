"""Context attribution on real news text (BASIL: Fox = context A, NYT = context B, HuffPost = an UNMODELED third context).
Bigram context models are trained on 80% of the events; every test sequence uses the held-out 20%. The background
'unknown' component U is trained on BABE's neutral US headlines (disjoint from BASIL).
Test sequences (n tokens each, truth known by construction):
  pure A / pure B      held-out Fox / NYT text (real)
  switching            alternating real spans of 30-60 tokens from Fox and NYT covering the same event
  unknown              held-out HuffPost text (real, unmodeled)
  half unknown         a real Fox span then a real HuffPost span
  blend(lam)           text generated token by token from lam p_A + (1 - lam) p_B (known lam; the only model-sampled case)
Reports: the decision confusion table, theta-band coverage on blends, mu bounds, and the switching threshold calibrated
under the blend null. Set BASIL_DIR and BABE_DIR."""
import os, sys, json, glob, time, numpy as np, pandas as pd
from collections import Counter
h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(h, "../.."))
from amb_vigneaux import attribution as AT

BASIL = os.environ.get("BASIL_DIR", "/tmp/ds/BASIL"); BABE = os.environ.get("BABE_DIR", "/tmp/ds/BABE")
N = int(os.environ.get("NTOK", 200)); REPS = int(os.environ.get("REPS", 40)); rng = np.random.default_rng(0)

def load():
    arts = [json.load(open(f)) for f in glob.glob(os.path.join(BASIL, "articles", "*", "*.json"))]
    by = {}
    for a in arts:
        txt = " ".join(" ".join(p) if isinstance(p, list) else p for p in a["body-paragraphs"])
        by.setdefault(a["triplet-uuid"], {})[a["source"].lower()] = AT.tokenize(txt)
    ev = sorted(e for e, d in by.items() if {"fox", "nyt", "hpo"} <= set(d)); r = np.random.default_rng(1); r.shuffle(ev)
    k = int(0.8 * len(ev)); return by, ev[:k], ev[k:]

if __name__ == "__main__":
    t0 = time.time(); by, tr, te = load()
    bg = [AT.tokenize(t) for t in pd.read_csv(os.path.join(BABE, "data", "news_headlines_usa_neutral.csv"))["title"].dropna().astype(str).tolist()[:40000]]
    cnt = Counter(w for e in tr for s in ("fox", "nyt") for w in by[e][s]); cnt.update(w for d in bg for w in d)
    vocab = sorted(w for w, c in cnt.items() if c >= 2) + ["<unk>"]; vs = set(vocab)
    M = lambda docs: AT.BigramLM([AT.map_unk(d, vs) for d in docs], vocab)
    LA, LB, LU = M([by[e]["fox"] for e in tr]), M([by[e]["nyt"] for e in tr]), M(bg)
    print("vocab %d, train events %d, test events %d, models built in %.0fs" % (len(vocab), len(tr), len(te), time.time() - t0), flush=True)
    probs = lambda toks: (LA.seq(toks), LB.seq(toks), LU.seq(toks))
    def window(src):
        e = te[rng.integers(len(te))]; t = AT.map_unk(by[e][src], vs)
        if len(t) <= N: t = (t * (N // max(len(t), 1) + 2))
        s = rng.integers(0, len(t) - N); return t[s:s + N]
    def switching():
        e = te[rng.integers(len(te))]; ta, tb = AT.map_unk(by[e]["fox"], vs), AT.map_unk(by[e]["nyt"], vs); out = []; src = rng.integers(2); ia = ib = 0
        while len(out) < N:
            L = int(rng.integers(30, 61)); t = ta if src == 0 else tb; i = ia if src == 0 else ib
            out += (t * 3)[i:i + L]; ia, ib = (ia + L, ib) if src == 0 else (ia, ib + L); src = 1 - src
        return out[:N]
    def half_unknown():
        return window("fox")[: N // 2] + window("hpo")[: N - N // 2]
    def blend(lam):
        prev, out = "<s>", []
        for _ in range(N):
            lm = LA if rng.random() < lam else LB; w, _ = lm.sample_next(prev, rng); out.append(w); prev = w
        return out
    # calibrate the switching threshold under the blend null (model-sampled blends), 95th percentile
    null = [AT.switching_stat(*probs(blend(lam))[:2]) for lam in (0.3, 0.5, 0.7) for _ in range(12)]
    crit = float(np.quantile(null, 0.95)); print("switching threshold (95%% under blend null, %d sims): %.2f" % (len(null), crit), flush=True)
    cases = {"pure A": lambda: window("fox"), "pure B": lambda: window("nyt"), "switching": switching,
             "unknown (HuffPost)": lambda: window("hpo"), "half unknown": half_unknown,
             "blend 0.2": lambda: blend(0.2), "blend 0.5": lambda: blend(0.5), "blend 0.8": lambda: blend(0.8)}
    rows = []
    for name, gen in cases.items():
        for _ in range(REPS):
            r = AT.attribute(*probs(gen()), switch_crit=crit); r["case"] = name; rows.append(r)
        sub = [r for r in rows if r["case"] == name]; dec = Counter(r["decision"] for r in sub)
        extra = ""
        if name.startswith("blend"):
            lam = float(name.split()[1]); cov = np.mean([r["theta_band"][0] <= lam <= r["theta_band"][1] for r in sub])
            extra = "| theta %.2f, band coverage %.2f, mean band width %.2f" % (np.mean([r["theta"] for r in sub]), cov, np.mean([r["theta_band"][1] - r["theta_band"][0] for r in sub]))
        print("%-20s decisions %s | theta %.2f | mu %.2f, mu lower bound %.2f %s" % (name, dict(dec), np.nanmean([r["theta"] for r in sub]), np.mean([r["mu"] for r in sub]), np.mean([r["mu_lower"] for r in sub]), extra), flush=True)
    json.dump(dict(N=N, reps=REPS, switch_crit=crit, rows=rows), open(os.path.join(h, "results_n%d.json" % N), "w"), default=float)
