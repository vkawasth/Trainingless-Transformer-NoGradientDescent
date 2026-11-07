"""kit_cycle_context.py -- Study 16 of PREREG_study16.md (DRAFT): a four-question cycle for LLM answers, measured against the marginal
projection of the companion note (sections 12.15-12.19).

  python kit_cycle_context.py items --out prompts16.jsonl                        # build the 1440 prompts (deterministic, seed 20261106)
  python kit_cycle_context.py run   --prompts prompts16.jsonl --model <hf id or path> --out answers16.jsonl [--resume] [--device mps]
                                                                                # local inference only (author's machine)
  python kit_cycle_context.py score --prompts prompts16.jsonl --answers answers16.jsonl --out study16_final.json
  python kit_cycle_context.py smoke                                              # simulated responders only, no language model

Design in one paragraph: each item has four fictitious gauges A0, A1, B0, B1, each reading 'high' or 'low'. The context gives M = 10
documents for each of the four cycle relations A0~B0, A0~B1, A1~B0 (stated 'same') and A1~B1 (stated 'different'); a fraction
(1 + lam)/2 of each relation's documents support it and the rest contradict it, so an ideal follower has correlations lam, lam, lam, -lam
(CHSH value S = 4 lam). The same documents are shown with six questions: the four cycle pairs, and the two pairs that no document relates,
(A0, A1) and (B0, B1). lam runs over 0, 0.2, 0.4, 0.6, 0.8, 1.0, which crosses the noncontextuality boundary S = 2 between 0.4 and 0.6.
"""
import json, argparse, random, itertools, os
import numpy as np
SEED, B, M, KSAMP, NITEMS = 20261106, 2000, 10, 20, 40
LAMS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CYCLE = [("A0", "B0", "same"), ("A0", "B1", "same"), ("A1", "B0", "same"), ("A1", "B1", "different")]
DIAG = [("A0", "A1"), ("B0", "B1")]
SAME = ["The {x} gauge and the {y} gauge always show the same reading.", "Whenever the {x} gauge reads high the {y} gauge reads high too, and likewise for low.",
        "Engineers note that the {x} and {y} gauges agree.", "The {y} gauge mirrors the {x} gauge."]
DIFF = ["The {x} gauge and the {y} gauge always show opposite readings.", "Whenever the {x} gauge reads high the {y} gauge reads low, and vice versa.",
        "Engineers note that the {x} and {y} gauges disagree.", "The {y} gauge shows the reverse of the {x} gauge."]

def name(rng):
    cons, vow = "bdfgklmnprstvz", "aeiou"; return "".join(rng.choice(cons) + rng.choice(vow) for _ in range(2)).capitalize() + rng.choice(["ik", "or", "an", "el"])

def items():
    rng = random.Random(SEED); out = []; pid = 0
    for i in range(NITEMS):
        names = {}
        while len(set(names.values())) < 4: names = {g: name(rng) for g in ("A0", "A1", "B0", "B1")}
        for lam in LAMS:
            r2 = random.Random(f"{SEED}-{i}-{lam}"); docs = []; nsup = int(round(M * (1 + lam) / 2))
            for (x, y, rel) in CYCLE:
                for j in range(M):
                    supports = j < nsup; says_same = (rel == "same") == supports
                    docs.append(r2.choice(SAME if says_same else DIFF).format(x=names[x], y=names[y]))
            r2.shuffle(docs); body = "Documents:\n" + "\n".join(f"- {d}" for d in docs)
            for kind, pair in [("cycle", (x, y)) for (x, y, _) in CYCLE] + [("diag", p) for p in DIAG]:
                o = ["high", "low"]; r2.shuffle(o)
                q = (f"According to the documents above, what are the current readings of the {names[pair[0]]} gauge and the {names[pair[1]]} gauge? "
                     f"Each reads {o[0]} or {o[1]}. Answer exactly in the form: {names[pair[0]]}: <reading>, {names[pair[1]]}: <reading>.")
                out.append(dict(id=pid, item=i, lam=lam, kind=kind, pair=list(pair), names=[names[pair[0]], names[pair[1]]], prompt=body + "\n\n" + q)); pid += 1
    return out

def parse(text, n1, n2):
    """(v1, v2) with v in {0 = high, 1 = low}, or None: for each gauge name, the first 'high' or 'low' within 25 characters after it"""
    t = text.lower(); out = []
    for n in (n1.lower(), n2.lower()):
        k = t.find(n)
        if k < 0: return None
        seg = t[k + len(n): k + len(n) + 25]; ih, il = seg.find("high"), seg.find("low")
        if ih < 0 and il < 0: return None
        out.append(0 if (il < 0 or (0 <= ih < il)) else 1)
    return tuple(out)

def run(prompts, model, out, resume=False, device=None):
    """local inference: same settings for every prompt (temperature 1, top_p 1, 16 new tokens, KSAMP samples, seed SEED + id); the prompt
    is read once and its cache shared by the samples; GPU (mps or cuda) in bfloat16 when available; each answer is flushed at once"""
    import torch, time
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev in ("mps", "cuda") else torch.float32
    done = set()
    if resume and os.path.exists(out):
        for line in open(out):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    elif os.path.exists(out) and os.path.getsize(out) > 0:
        raise SystemExit(f"{out} already has answers; use --resume to continue or choose a new --out")
    todo = [p for p in prompts if p["id"] not in done]
    print(f"device {dev}; {len(done)} already done, {len(todo)} to run", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    kw = dict(attn_implementation="eager") if dev == "mps" else {}
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kw).to(dev); m.eval(); base = getattr(m, "model", m); t0 = time.time()
    with open(out, "a" if resume else "w") as fh:
        for n, p in enumerate(todo, 1):
            torch.manual_seed(SEED + p["id"])
            ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                o_ = base(ids[:, :-1], use_cache=True); cache = o_.past_key_values; del o_; cache.batch_repeat_interleave(KSAMP)
                many = ids.repeat(KSAMP, 1)
                g = m.generate(many, attention_mask=torch.ones_like(many), past_key_values=cache, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=16, num_return_sequences=1, pad_token_id=tok.eos_token_id)
            texts = [tok.decode(x[ids.shape[1]:], skip_special_tokens=True) for x in g]
            fh.write(json.dumps(dict(id=p["id"], samples=texts)) + "\n"); fh.flush()
            del g, ids, cache
            if dev == "mps": torch.mps.empty_cache()
            el = time.time() - t0; print(f"\r{len(done) + n}/{len(prompts)}  {el / n:.1f}s per prompt  ETA {el / n * (len(todo) - n) / 60:.0f} min   ", end="", flush=True)
    print("\ndone", flush=True)

# ------------------------------------------------------------------ analysis
CELLS = list(itertools.product((0, 1), repeat=4)); GIDX = {"A0": 0, "A1": 1, "B0": 2, "B1": 3}

def box_interval(boxes, target):
    """[min, max] of P(target pair equal) over joint laws on 16 cells whose cycle-pair cells lie in the given boxes; None if infeasible"""
    from scipy.optimize import linprog
    Aub, bub = [], []
    for (x, y, _) in CYCLE:
        i, j = GIDX[x], GIDX[y]
        for a in (0, 1):
            for b in (0, 1):
                row = [1.0 if (c[i] == a and c[j] == b) else 0.0 for c in CELLS]; lo, hi = boxes[(x, y)][a][b]
                Aub.append(row); bub.append(hi); Aub.append([-v for v in row]); bub.append(-lo)
    i, j = GIDX[target[0]], GIDX[target[1]]; w = np.array([1.0 if c[i] == c[j] else 0.0 for c in CELLS])
    kw = dict(A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.ones((1, 16)), b_eq=[1.0], bounds=[(0, None)] * 16, method="highs")
    r1 = linprog(w, **kw); r2 = linprog(-w, **kw)
    if r1.status != 0 or r2.status != 0: return None
    return [float(r1.fun), float(-r2.fun)]

def score(prompts, answers):
    """counts C[item, lam, question, a, b] (questions: the four cycle pairs, then the two unasked pairs), split-half counts for the
    noise baseline, and invalid rates per (lam, kind)"""
    A = {a["id"]: a["samples"] for a in answers}; Q = [(x, y) for (x, y, _) in CYCLE] + DIAG
    C = np.zeros((NITEMS, len(LAMS), 6, 2, 2)); H = np.zeros((2, NITEMS, len(LAMS), 6, 2, 2)); inval = {}; rr = random.Random(SEED + 1)
    for p in prompts:
        if p["id"] not in A: continue
        li, qi = LAMS.index(p["lam"]), Q.index(tuple(p["pair"])); key = f"{p['lam']}|{p['kind']}"; inval.setdefault(key, [0, 0])
        vs = []
        for s_ in A[p["id"]]:
            v = parse(s_, *p["names"]); inval[key][1] += 1
            if v is None: inval[key][0] += 1; continue
            vs.append(v); C[p["item"], li, qi, v[0], v[1]] += 1
        rr.shuffle(vs)
        for k, v in enumerate(vs): H[k % 2, p["item"], li, qi, v[0], v[1]] += 1
    return dict(C=C, H=H, invalid={k: c[0] / c[1] for k, c in inval.items()})

def _stats(T):
    """T[lam, question, a, b] pooled counts -> per lam: correlations, S, CbD statistic, drift, unasked-pair agreement"""
    out = []
    for li in range(len(LAMS)):
        t = T[li]; n = t.sum(axis=(1, 2))
        if np.any(n[:4] == 0): out.append(None); continue
        e = [(t[q, 0, 0] + t[q, 1, 1] - t[q, 0, 1] - t[q, 1, 0]) / n[q] for q in range(4)]
        mA = {}; 
        for q, (x, y, _) in enumerate(CYCLE):
            mA.setdefault(x, []).append((t[q, 0, :].sum() - t[q, 1, :].sum()) / n[q]); mA.setdefault(y, []).append((t[q, :, 0].sum() - t[q, :, 1].sum()) / n[q])
        delta = sum(abs(v[0] - v[1]) for v in mA.values())
        s_odd = max(sum(s * v for s, v in zip(sg, e)) for sg in itertools.product((1, -1), repeat=4) if np.prod(sg) == -1)
        el = [None if n[4 + k] == 0 else float((t[4 + k, 0, 0] + t[4 + k, 1, 1]) / n[4 + k]) for k in range(2)]
        cells = [(t[q] / n[q]) for q in range(4)]
        out.append(dict(S=float(e[0] + e[1] + e[2] - e[3]), cbd=float(s_odd - 2 - delta), drift=float(delta / 4), elicited=el, cells=cells))
    return out

def analyse(sc):
    C, H = sc["C"], sc["H"]; rng = np.random.default_rng(SEED); n = C.shape[0]
    est = _stats(C.sum(axis=0)); boots = []; excess = []
    def half_drift(Hw):                                   # the drift statistic computed between two random halves of the same context
        a, b = Hw[0], Hw[1]; vals = []
        for li in range(len(LAMS)):
            for q in range(4):
                na, nb = a[li, q].sum(), b[li, q].sum()
                if na and nb: vals.append(abs((a[li, q, 0, :].sum() - a[li, q, 1, :].sum()) / na - (b[li, q, 0, :].sum() - b[li, q, 1, :].sum()) / nb))
        return float(np.mean(vals))
    for _ in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        st = _stats(np.einsum("i,ilqab->lqab", w, C)); boots.append(st)
        dr = np.mean([s_["drift"] for s_ in st if s_]); excess.append(dr - half_drift(np.einsum("i,hilqab->hlqab", w, H)))
    res = dict(per_lambda={}, invalid=sc["invalid"])
    for li, lam in enumerate(LAMS):
        if est[li] is None: continue
        bl = [b[li] for b in boots if b[li] is not None]
        ci = lambda f: [float(np.percentile([f(b) for b in bl], 2.5)), float(np.percentile([f(b) for b in bl], 97.5))]
        boxes = {(x, y): [[[float(np.percentile([b["cells"][q][a][bb] for b in bl], 2.5)), float(np.percentile([b["cells"][q][a][bb] for b in bl], 97.5))]
                           for bb in (0, 1)] for a in (0, 1)] for q, (x, y, _) in enumerate(CYCLE)}
        iv = [box_interval(boxes, p) for p in DIAG]; el = est[li]["elicited"]
        coh = [None if (iv[k] is None or el[k] is None) else bool(iv[k][0] <= el[k] <= iv[k][1]) for k in range(2)]
        res["per_lambda"][str(lam)] = dict(S=est[li]["S"], S_ci=ci(lambda b: b["S"]), cbd=est[li]["cbd"], cbd_ci=ci(lambda b: b["cbd"]),
                                           drift=est[li]["drift"], identified={"A0A1": iv[0], "B0B1": iv[1]}, elicited={"A0A1": el[0], "B0B1": el[1]},
                                           coherent={"A0A1": coh[0], "B0B1": coh[1]})
    pl = res["per_lambda"]
    res["H1"] = dict(drift=float(np.mean([v["drift"] for v in pl.values()])), noise_baseline=half_drift(H),
                     ci_excess=[float(np.percentile(excess, 2.5)), float(np.percentile(excess, 97.5))])
    res["H1"]["supported"] = bool(res["H1"]["ci_excess"][0] > 0)
    top, bot = pl.get("1.0"), pl.get("0.0")
    res["H2"] = dict(cbd_ci_at_1=top and top["cbd_ci"], cbd_ci_at_0=bot and bot["cbd_ci"], supported=bool(top and bot and top["cbd_ci"][0] > 0 and bot["cbd_ci"][1] < 0))
    feas = [c for v in pl.values() for c in v["coherent"].values() if c is not None]
    res["H3"] = dict(feasible_cases=len(feas), coherent_cases=int(sum(feas)), supported=(None if not feas else bool(all(feas))))
    return res

def smoke():
    """two simulated responders, no language model: (a) a 'follower' that answers each cycle question from that relation's documents alone
    (correlation lam or -lam, uniform marginals) and each unasked pair independently; (b) a 'drifter' that also tilts each gauge toward
    'high' by 0.15 when it is named first"""
    P = items(); out = {}
    for label, tilt in (("follower", 0.0), ("drifter", 0.15)):
        rng = np.random.default_rng(1); ans = []
        for p in P:
            lam = p["lam"]; samples = []
            for _ in range(KSAMP):
                if p["kind"] == "cycle":
                    rel = [r for (x, y, r) in CYCLE if [x, y] == p["pair"]][0]; same = rng.random() < ((1 + lam) / 2 if rel == "same" else (1 - lam) / 2)
                    a = 0 if rng.random() < 0.5 + tilt else 1; b = a if same else 1 - a
                else: a = int(rng.random() < 0.5); b = int(rng.random() < 0.5)
                w = ["high", "low"]; samples.append(f"{p['names'][0]}: {w[a]}, {p['names'][1]}: {w[b]}")
            ans.append(dict(id=p["id"], samples=samples))
        r = analyse(score(P, ans))
        out[label] = dict(H1=r["H1"]["supported"], H2=r["H2"]["supported"], H3=r["H3"],
                          S={k: round(v["S"], 3) for k, v in r["per_lambda"].items()}, cbd={k: round(v["cbd"], 3) for k, v in r["per_lambda"].items()})
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--out"); ap.add_argument("--prompts"); ap.add_argument("--answers")
    ap.add_argument("--model"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--device")
    a = ap.parse_args()
    if a.cmd == "items":
        with open(a.out, "w") as fh:
            for p in items(): fh.write(json.dumps(p) + "\n")
    elif a.cmd == "run": run([json.loads(l) for l in open(a.prompts)], a.model, a.out, a.resume, a.device)
    elif a.cmd == "score":
        json.dump(analyse(score([json.loads(l) for l in open(a.prompts)], [json.loads(l) for l in open(a.answers)])), open(a.out, "w"), indent=1)
    elif a.cmd == "smoke": smoke()
