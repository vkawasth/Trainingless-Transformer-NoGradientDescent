"""Document-context attribution (the retrieval-augmented setting): two specific documents A and B sit in the context
window; generated tokens may copy from A, from B, blend the two token by token, switch between them in spans, or come
from a third document that is not in the context (unknown). Real BASIL articles; background language model trained on
the training articles plus BABE neutral headlines. Context model = 0.5 doc bigram + 0.5 background.
Generated text (n tokens, truth known):
  copy A / copy B         a real span of the document, with 20% of tokens replaced by background samples (paraphrase noise)
  switching               alternating real spans (20-50 tokens) of A and B, same noise
  blend(lam)              sampled token by token from lam p_A + (1 - lam) p_B
  unknown                 a real span of a third document (not in the context), same noise
  half unknown            first half copied from A, second half from the third document
Pairs: 'hard' = A and B are two outlets' articles on the SAME event; 'easy' = articles on different events."""
import os, sys, json, glob, time, numpy as np, pandas as pd
from collections import Counter
h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(h, "../.."))
from amb_vigneaux import attribution as AT
from run import load, BASIL, BABE

ARCH = os.environ.get("ARCH", "docs")   # "docs": decompose on (doc A, doc B, background) -- affinely independent directions
N = int(os.environ.get("NTOK", 100)); REPS = int(os.environ.get("REPS", 30)); NOISE = float(os.environ.get("NOISE", 0.2)); rng = np.random.default_rng(0)

if __name__ == "__main__":
    t0 = time.time(); by, tr, te = load()
    bg = [AT.tokenize(t) for t in pd.read_csv(os.path.join(BABE, "data", "news_headlines_usa_neutral.csv"))["title"].dropna().astype(str).tolist()[:40000]]
    bg += [by[e][s] for e in tr for s in ("fox", "nyt", "hpo")]
    cnt = Counter(w for d in bg for w in d); cnt.update(w for e in te for s in ("fox", "nyt", "hpo") for w in by[e][s])
    vocab = sorted(w for w, c in cnt.items() if c >= 2) + ["<unk>"]; vs = set(vocab)
    BG = AT.BigramLM([AT.map_unk(d, vs) for d in bg], vocab)
    docs = [(e, s, AT.map_unk(by[e][s], vs)) for e in te for s in ("fox", "nyt", "hpo")]
    print("vocab %d, %d test documents, background built in %.0fs" % (len(vocab), len(docs), time.time() - t0), flush=True)
    def noisy(span, prev="<s>"):
        out = []
        for w in span:
            if rng.random() < NOISE: w, _ = BG.sample_next(prev, rng)
            out.append(w); prev = w
        return out
    def span(t, L):
        t = t if len(t) > L + 1 else t * (L // max(len(t), 1) + 2); s = rng.integers(0, len(t) - L); return t[s:s + L]
    def pick(hard):
        e = te[rng.integers(len(te))]
        if hard:
            sa, sb = rng.choice(["fox", "nyt", "hpo"], 2, replace=False); A, B = by[e][sa], by[e][sb]
        else:
            e2 = te[(te.index(e) + 1 + rng.integers(len(te) - 1)) % len(te)]; A, B = by[e]["fox"], by[e2]["nyt"]
        others = [d for (ee, s, d) in docs if ee != e and (hard or ee != e2)]
        C = others[rng.integers(len(others))]
        return [AT.map_unk(x, vs) for x in (A, B)] + [C]
    def make(case, A, B, C, lam=None):
        if case == "copy A": return noisy(span(A, N))
        if case == "copy B": return noisy(span(B, N))
        if case == "unknown": return noisy(span(C, N))
        if case == "half unknown": return noisy(span(A, N // 2)) + noisy(span(C, N - N // 2))
        if case == "switching":
            out, src = [], int(rng.integers(2))
            while len(out) < N: out += noisy(span(A if src == 0 else B, int(rng.integers(20, 51)))); src = 1 - src
            return out[:N]
        if case.startswith("blend"):
            la, lb = AT.ContextLM(A, BG, vocab, beta=0.8), AT.ContextLM(B, BG, vocab, beta=0.8); prev, out = "<s>", []
            for _ in range(N):
                w, _ = (la if rng.random() < lam else lb).sample_next(prev, rng); out.append(w); prev = w
            return out
    results = {}
    for hard in (False, True):
        tag = "same-event pair (hard)" if hard else "different-event pair (easy)"
        null = []
        for _ in range(30):
            A, B, C = pick(hard); la, lb = AT.ContextLM(A, BG, vocab), AT.ContextLM(B, BG, vocab)
            toks = make("blend", A, B, C, lam=rng.choice([0.3, 0.5, 0.7])); null.append(AT.switching_stat(la.seq(toks), lb.seq(toks)))
        crit = float(np.quantile(null, 0.95))
        def comps(A, B, toks):
            la, lb = AT.ContextLM(A, BG, vocab), AT.ContextLM(B, BG, vocab)
            if ARCH == "docs": return la.doc.seq(toks), lb.doc.seq(toks), BG.seq(toks), la.seq(toks), lb.seq(toks)
            return la.seq(toks), lb.seq(toks), BG.seq(toks), la.seq(toks), lb.seq(toks)
        # calibrate the unknown budget: 95th percentile of the background share's lower bound on known-context text
        cal = []
        for case in ("copy A", "copy B", "switching", "blend 0.5"):
            for _ in range(8):
                A, B, C = pick(hard); toks = make(case, A, B, C, lam=0.5); a, b, u, _, _ = comps(A, B, toks)
                cal.append(AT.attribute(a, b, u)["mu_lower"])
        budget = float(np.quantile(cal, 0.95))
        print("\n%s: switching threshold %.2f, unknown budget (background share above which = unknown) %.2f" % (tag, crit, budget), flush=True)
        rows = []
        for case in ("copy A", "copy B", "switching", "blend 0.2", "blend 0.5", "blend 0.8", "unknown", "half unknown"):
            for _ in range(REPS):
                A, B, C = pick(hard); la, lb = AT.ContextLM(A, BG, vocab), AT.ContextLM(B, BG, vocab)
                lam = float(case.split()[1]) if case.startswith("blend") else None
                toks = make(case, A, B, C, lam); a, b, u, fa, fb = comps(A, B, toks)
                r = AT.attribute(a, b, u, mu_budget=budget); r["switch_stat"] = AT.switching_stat(fa, fb)
                if r["decision"] != "unknown" and r["switch_stat"] > crit: r["decision"] = "switching"
                r["case"] = case; rows.append(r)
            sub = [r for r in rows if r["case"] == case]; dec = Counter(r["decision"] for r in sub); extra = ""
            if lam is not None:
                extra = "| band coverage %.2f, width %.2f" % (np.mean([r["theta_band"][0] <= lam <= r["theta_band"][1] for r in sub]), np.mean([r["theta_band"][1] - r["theta_band"][0] for r in sub]))
            print("  %-13s %s | theta %.2f | mu %.2f (lower bound %.2f) %s" % (case, dict(dec), np.nanmean([r["theta"] for r in sub]), np.mean([r["mu"] for r in sub]), np.mean([r["mu_lower"] for r in sub]), extra), flush=True)
        results[tag] = dict(crit=crit, budget=budget, rows=rows)
    json.dump(dict(N=N, reps=REPS, noise=NOISE, results=results), open(os.path.join(h, "docs_%s_n%d.json" % (ARCH, N)), "w"), default=float)
