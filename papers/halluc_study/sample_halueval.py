"""sample_halueval.py -- Study 2, step 2: draw K = 5 samples per HaluEval prompt at temperature 1.0 (as pre-registered).

Backends (pick one with --backend):
  anthropic   Claude via the Anthropic API.      pip install anthropic   ; export ANTHROPIC_API_KEY=...
  openai      OpenAI-compatible chat API.        pip install openai      ; export OPENAI_API_KEY=...
              (also works for any OpenAI-compatible server, e.g. vLLM, via --base-url)
  ollama      a local model served by Ollama.    https://ollama.com      ; ollama pull llama3.1
  mock        no model; returns canned answers (for testing the pipeline)

Examples:
  python sample_halueval.py --backend anthropic --model claude-haiku-4-5-20251001 --limit 200
  python sample_halueval.py --backend openai    --model gpt-4o-mini --limit 200
  python sample_halueval.py --backend ollama    --model llama3.1 --workers 2
  python sample_halueval.py --backend mock      --limit 50

Output: samples.jsonl, one line {"id": ..., "samples": [5 strings]} per item. The script resumes: items already in the output file are
skipped, so you can stop and restart, or grow --limit later. Then score with:
  python kit_halueval.py score --data qa_data.json --samples samples.jsonl --out study2_results.json
"""
import argparse, json, os, sys, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed

K, TEMPERATURE = 5, 1.0                      # fixed by the pre-registration
SYSTEM = "Answer with a short phrase only, no explanation."
MAX_TOKENS = 64

def make_backend(args):
    if args.backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        def gen(prompt):
            r = client.messages.create(model=args.model, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, system=SYSTEM,
                                       messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
        return gen
    if args.backend == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
        def gen(prompt):
            r = client.chat.completions.create(model=args.model, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                                               messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            return (r.choices[0].message.content or "").strip()
        return gen
    if args.backend == "ollama":
        import urllib.request
        url = (args.base_url or "http://localhost:11434") + "/api/chat"
        opts = {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": args.num_ctx}
        if args.cpu: opts["num_gpu"] = 0          # run on CPU only: slower, but avoids GPU (Metal/CUDA) compute errors
        def gen(prompt):
            body = json.dumps({"model": args.model, "stream": False, "keep_alive": "30m", "options": opts,
                               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            if "error" in data: raise RuntimeError(f"ollama error: {data['error']}")      # never store an error as a sample
            text = (data.get("message") or {}).get("content", "").strip()
            if not text: raise RuntimeError("empty answer from ollama")
            return text
        return gen
    if args.backend == "mock":
        rng = random.Random(0)
        def gen(prompt):
            return rng.choice(["Arthur's Magazine", "First for Women", "Delhi", "1844", "yes", "no"])
        return gen
    sys.exit(f"unknown backend {args.backend}")

def with_retries(gen, prompt, tries=8):
    for k in range(tries):
        try:
            return gen(prompt)
        except Exception as e:              # rate limits, timeouts, local GPU compute errors: back off and retry
            if k == tries - 1: raise
            wait = min(120, 3 * 2 ** k) + random.random()
            print(f"  retry in {wait:.1f}s after: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr); time.sleep(wait)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["anthropic", "openai", "ollama", "mock"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--out", default="samples.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (try 200 first)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests; lower it if you hit rate limits (use 1 for ollama)")
    ap.add_argument("--num-ctx", type=int, default=1024, help="ollama context window; prompts are ~150 tokens, so 1024 saves memory")
    ap.add_argument("--cpu", action="store_true", help="ollama: run on CPU only (use if the GPU keeps raising compute errors)")
    args = ap.parse_args()
    if args.backend != "mock" and not args.model: sys.exit("--model is required")
    gen = make_backend(args)

    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
    if args.limit: prompts = prompts[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(done)} already sampled, {len(todo)} to go, K={K}, temperature={TEMPERATURE}, backend={args.backend} {args.model}")

    def job(p): return {"id": p["id"], "samples": [with_retries(gen, p["prompt"]) for _ in range(K)]}
    t0 = time.time()
    with open(args.out, "a") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, p) for p in todo]
        failed = 0
        for n, f in enumerate(as_completed(futs), 1):
            try:
                out.write(json.dumps(f.result()) + "\n"); out.flush()
            except Exception as e:          # leave the item out; rerunning the script retries it
                failed += 1; print(f"  item skipped after retries: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} items  ({(time.time() - t0) / n:.1f}s per item)")
    print(f"wrote {args.out}" + (f"; {failed} items failed and will be retried on the next run" if todo and failed else ""))

if __name__ == "__main__":
    main()
