"""Context attribution on BASIL: contexts are word-bigram language models trained on Fox (A) and on NYT (B), on 70 training
events. The unknown/background U is a bigram model on all three outlets' training text. Test generations are sampled
from bigram models with KNOWN provenance:
  A, B                 pure context
  mix(lam)             each token from A with prob lam, else B (token-level interleaving)
  switch               alternating blocks of 40 tokens A / B (segment switching)
  unknown              a model trained on the 30 HELD-OUT events, all outlets (new topics)
  contam(mu)           token-level: unknown with prob mu, else A
Lengths 100 and 400 tokens, 30 sequences per scenario. Reported: category accuracy, 95% band coverage of lam_A (mixtures),
unknown lower bound > 0 rate (detection power; false-positive rate on clean text), single-source posterior.
Needs BASIL_DIR (the BASIL repository root)."""
import os, sys, re, json, glob, collections, numpy as np
h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(h, "../.."))
from amb_vigneaux import attribution as AT

TOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")

def load(root):
    arts = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(root, "articles", "*", "*.json")):
        d = json.load(open(f)); text = " ".join(d.get("body-paragraphs") and sum(d["body-paragraphs"], []) if isinstance(d["body-paragraphs"][0], list) else d["body-paragraphs"])
        arts[d["triplet-uuid"]][d["source"].lower()] = TOK.findall(text.lower())
    return {k: v for k, v in arts.items() if {"fox", "nyt", "hpo"} <= set(v)}

class Bigram:
    def __init__(self, docs, vocab, alpha=0.6):
        self.V = vocab; self.ix = {w: i for i, w in enumerate(vocab)}; n = len(vocab); self.alpha = alpha
        uni = np.full(n, 0.5); self.big = collections.defaultdict(collections.Counter)
        for d in docs:
            prev = "<s>"
            for w in d:
                uni[self.ix[w]] += 1; self.big[prev][w] += 1; prev = w
        self.uni = uni / uni.sum(); self.tot = {v: sum(c.values()) for v, c in self.big.items()}
    def p(self, prev, w):
        c = self.big.get(prev); pu = self.uni[self.ix[w]]
        if not c: return pu
        return self.alpha * c.get(w, 0) / self.tot[prev] + (1 - self.alpha) * pu
    def dist(self, prev):
        d = (1 - self.alpha) * self.uni.copy(); c = self.big.get(prev)
        if c:
            for w, k in c.items(): d[self.ix[w]] += self.alpha * k / self.tot[prev]
        return d / d.sum()

def generate(models, chooser, n, rng):
    out, prev = [], "<s>"
    for i in range(n):
        m = models[chooser(i)]; w = m.V[rng.choice(len(m.V), p=m.dist(prev))]; out.append(w); prev = w
    return out

def probs(seq, models):
    prev = "<s>"; P = []
    for w in seq: P.append([m.p(prev, w) for m in models]); prev = w
    return np.array(P)

if __name__ == "__main__":
    root = os.environ.get("BASIL_DIR", "/tmp/ds/BASIL"); arts = load(root); keys = sorted(arts); rng = np.random.default_rng(0)
    rng.shuffle(keys); tr, te = keys[:70], keys[70:]
    vocab = sorted({w for k in keys for s in arts[k].values() for w in s} | {"<s>"})
    A = Bigram([arts[k]["fox"] for k in tr], vocab); B = Bigram([arts[k]["nyt"] for k in tr], vocab)
    U = Bigram([arts[k][s] for k in tr for s in ("fox", "nyt", "hpo")], vocab)
    C = Bigram([arts[k][s] for k in te for s in ("fox", "nyt", "hpo")], vocab)            # unknown: new topics
    gens = {"A": ({"A": A}, lambda i, r: "A", "single:A"), "B": ({"B": B}, lambda i, r: "B", "single:B"),
            "mix(0.2)": (None, 0.2, "mix"), "mix(0.5)": (None, 0.5, "mix"), "mix(0.8)": (None, 0.8, "mix"),
            "switch": ({"A": A, "B": B}, lambda i, r: "AB"[(i // 40) % 2], "switch"),
            "unknown": ({"C": C}, lambda i, r: "C", "single:U"), "contam(0.2)": (None, 0.2, "contam"), "contam(0.4)": (None, 0.4, "contam")}
    rows = []; NREP = int(os.environ.get("NREP", 30))
    for n in (100, 400):
        for name, spec in gens.items():
            for rep in range(NREP):
                r = np.random.default_rng(1000 * n + 17 * rep + hash(name) % 997)
                if name.startswith("mix"):
                    lam = spec[1]; flips = r.random(n) < lam; seq = generate({"A": A, "B": B}, lambda i: "A" if flips[i] else "B", n, r); truth = "mix"; true_lamA = lam
                elif name.startswith("contam"):
                    mu = spec[1]; flips = r.random(n) < mu; seq = generate({"A": A, "C": C}, lambda i: "C" if flips[i] else "A", n, r); truth = "contam"; true_lamA = 1 - mu
                else:
                    seq = generate(spec[0], lambda i, s=spec[1]: s(i, r), n, r); truth = spec[2]; true_lamA = {"single:A": 1.0, "single:B": 0.0}.get(truth, 0.5 if truth == "switch" else None)
                P = probs(seq, [A, B, U]); out = AT.read(P)
                lamv = np.array([out["lam"][k] for k in ("A", "B", "U")]); prev = "<s>"; dists = []; ids = []
                for w in seq: dists.append(np.vstack([M.dist(prev) for M in (A, B, U)])); ids.append(A.ix[w]); prev = w
                Hh, Vv, logm = AT.mixture_moments(dists, lamv); z = AT.surprise_z([-lm[i] for lm, i in zip(logm, ids)], Hh, Vv)
                out["Z"] = z
                if z > 3: out["category"] = "unknown" if out["category"].startswith("single:U") or name == "x" else out["category"] + "+unknown"
                rows.append(dict(n=n, scenario=name, truth=truth, true_lamA=true_lamA, **{k: out[k] for k in ("category", "lam", "bands", "unknown_lower", "single_posterior", "Z")}))
            sub = [x for x in rows if x["n"] == n and x["scenario"] == name]
            cats = collections.Counter(x["category"] for x in sub)
            cov = np.mean([x["bands"]["A"][0] <= x["true_lamA"] <= x["bands"]["A"][1] for x in sub]) if sub[0]["true_lamA"] is not None and name.startswith("mix") else float("nan")
            print("n=%d %-12s categories %s | lamA %.2f band [%.2f, %.2f] cover %.2f | unknown>0 %.2f | P(A|seq) %.2f | Z %.1f, Z>3 %.2f" % (
                n, name, dict(cats), np.mean([x["lam"]["A"] for x in sub]), np.mean([x["bands"]["A"][0] for x in sub]), np.mean([x["bands"]["A"][1] for x in sub]),
                cov, np.mean([x["unknown_lower"] > 0 for x in sub]), np.mean([x["single_posterior"]["A"] for x in sub]), np.mean([x["Z"] for x in sub]), np.mean([x["Z"] > 3 for x in sub])), flush=True)
    json.dump(rows, open(os.path.join(h, "results.json"), "w"), indent=0, default=float)
