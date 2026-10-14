"""kit_halueval.py -- Study 2 of PREREG_hallucination.md (needs your model; nothing here calls a model).

HaluEval QA (Li et al., EMNLP 2023; MIT licence): each item has knowledge, a question, a right answer and a hallucinated answer.

  python kit_halueval.py prepare  --data qa_data.json --out prompts.jsonl
      writes one prompt per item (knowledge + question); sample your model K = 5 times at temperature 1.0 and save
      samples.jsonl with lines {"id": <int>, "samples": [<str>, ...]}.
  python kit_halueval.py score    --data qa_data.json --samples samples.jsonl --out study2_results.json
      scores the right and the hallucinated answer of every item by the disagreement detector (fraction of samples that do not
      match the answer under the built-in normalised-containment judge) and reports the AUROC (hallucinated = 1) with a 95% bootstrap
      CI (B = 2000, seed 20261013), as pre-registered.
  python kit_halueval.py smoke    runs the pipeline on synthetic samples (no model, no data download needed).
"""
import json, re, sys, argparse
import numpy as np

SEED, B = 20261013, 2000
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()
def match(answer, sample):
    a, s = norm(answer), norm(sample)
    return bool(a) and bool(s) and (a in s or s in a)
def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int); pos, neg = score[y == 1], score[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))
def load_items(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def prepare(data, out):
    with open(out, "w") as f:
        for i, it in enumerate(load_items(data)):
            prompt = f"Answer the question using the knowledge.\nKnowledge: {it['knowledge']}\nQuestion: {it['question']}\nAnswer:"
            f.write(json.dumps({"id": i, "prompt": prompt}) + "\n")

def score_items(items, samples):
    S, Y = [], []
    for i, it in enumerate(items):
        if i not in samples or not samples[i]: continue
        for ans, lab in ((it["right_answer"], 0), (it["hallucinated_answer"], 1)):
            S.append(sum(not match(ans, s) for s in samples[i]) / len(samples[i])); Y.append(lab)
    S, Y = np.array(S), np.array(Y); a = auroc(S, Y); rng = np.random.default_rng(SEED)
    n = len(S) // 2; pairs = np.arange(n)
    bs = []
    for _ in range(B):
        ix = rng.choice(pairs, n); idx = np.concatenate([2 * ix, 2 * ix + 1]); bs.append(auroc(S[idx], Y[idx]))
    return dict(items=int(n), auroc=float(a), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

def smoke():
    rng = np.random.default_rng(1); items, samples = [], {}
    for i in range(300):
        right, wrong = f"answer {i}", f"other {i}"
        items.append({"right_answer": right, "hallucinated_answer": wrong})
        samples[i] = [right if rng.random() < 0.7 else (wrong if rng.random() < 0.3 else f"noise {i}") for _ in range(5)]
    r = score_items(items, samples); print("smoke:", r); assert r["auroc"] > 0.8, "smoke test failed"
    print("smoke test passed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--data"); ap.add_argument("--samples"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "prepare": prepare(a.data, a.out)
    elif a.cmd == "score":
        items = load_items(a.data); samples = {d["id"]: d["samples"] for d in load_items(a.samples)}
        r = score_items(items, samples); json.dump(r, open(a.out, "w"), indent=1); print(r)
    elif a.cmd == "smoke": smoke()
