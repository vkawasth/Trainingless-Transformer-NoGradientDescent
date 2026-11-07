"""run_icl_mps.py -- Deviation 2 runner for Study 15 (kit_icl_poison.py is unchanged and still frozen).

Same prompts (the kit's items()), same sampling settings (temperature 1, top_p 1, 5 new tokens, KSAMP samples, seed SEED + id),
same output format, so `python kit_icl_poison.py score ...` works on its output unchanged. Differences, declared as Deviation 2:
  * the model runs on the Apple GPU (mps) at half precision (bfloat16, or float16 if bfloat16 is unavailable), or on CUDA if present;
    sampled tokens therefore need not match a CPU fp32 run token for token
  * every answer is written and flushed immediately, with a progress line and an ETA
  * the prompt is read once per prompt and its cache shared by the KSAMP samples (same sampling distribution, faster);
    --no-share falls back to the kit's single generate call, and the script falls back by itself if this transformers lacks it
  * --resume skips ids already present in the output file (appends instead of overwriting)
  v3 (Deviation 2b, 2026-11-06; defaults leave the Qwen runs unchanged):
  * eager attention only for models with grouped-query heads on mps (where the fused kernel fails); other models (e.g. SmolLM2, one
    key/value head per query head) use the default fused attention
  * --chunk c draws the KSAMP samples c at a time from copies of the shared prompt cache, so only c copies exist at once (the
    random stream continues across chunks from the same per-prompt seed; same distribution, not token-identical to an unchunked run)
  * refuses to overwrite a non-empty output file unless --resume is given

  python run_icl_mps.py --model Qwen/Qwen2.5-1.5B-Instruct --out answers.jsonl            # full run, fresh file
  python run_icl_mps.py --model Qwen/Qwen2.5-1.5B-Instruct --out answers.jsonl --resume   # continue a stopped run
  python run_icl_mps.py --model Qwen/Qwen2.5-1.5B-Instruct --out test5.jsonl --limit 5    # quick timing test
  python run_icl_mps.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --out answers_smol2.jsonl --chunk 2   # secondary model
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kit_icl_poison as K

def gen_shared(m, tok, ids, torch, chunk=None):
    """Read the prompt once, then draw the KSAMP samples from copies of that cache. generate() still does the sampling, so the
    model's generation config (e.g. its top_k and repetition penalty) is applied exactly as in the plain call. With chunk = c the samples
    are drawn c at a time, each chunk from a fresh copy of the prompt cache."""
    import copy
    with torch.no_grad():
        base = getattr(m, "model", m)                   # the decoder without its output head: the same cache, no logits for every
        out = base(ids[:, :-1], use_cache=True)         # prompt position (those were about 0.5-1 GB per long prompt)
        cache0 = out.past_key_values; del out
        sizes = [K.KSAMP] if not chunk else [min(chunk, K.KSAMP - i) for i in range(0, K.KSAMP, chunk)]
        rows = []
        for c in sizes:
            cache = cache0 if len(sizes) == 1 else copy.deepcopy(cache0)
            cache.batch_repeat_interleave(c); many = ids.repeat(c, 1)
            g = m.generate(many, attention_mask=torch.ones_like(many), past_key_values=cache, do_sample=True, temperature=1.0, top_p=1.0,
                           max_new_tokens=5, num_return_sequences=1, pad_token_id=tok.eos_token_id)
            rows += [x for x in g]; del cache, g
        return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None, help="mps, cuda or cpu (default: best available)")
    ap.add_argument("--no-share", action="store_true", help="process the prompt once per sample (slower; plain generate)")
    ap.add_argument("--chunk", type=int, default=None, help="draw the samples this many at a time (less memory)")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if dev in ("mps", "cuda"):
        dtype = torch.bfloat16
        try: torch.zeros(1, dtype=dtype, device=dev) @ torch.zeros(1, dtype=dtype, device=dev)
        except Exception: dtype = torch.float16
    prompts = K.items()[: a.limit] if a.limit else K.items()
    done = set()
    if a.resume and os.path.exists(a.out):
        with open(a.out) as fh:
            for line in fh:
                try: done.add(json.loads(line)["id"])
                except Exception: pass                      # a half-written last line from a hard kill is ignored and redone
    if not a.resume and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        raise SystemExit(f"{a.out} already has answers: add --resume to continue it, or choose a new --out")
    todo = [p for p in prompts if p["id"] not in done]
    print(f"device {dev}, dtype {str(dtype).split('.')[-1]}; {len(done)} already done, {len(todo)} to run", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    # eager attention only where needed: PyTorch's fused attention on mps fails on grouped-query heads (12 query vs 2 key/value heads
    # in Qwen2.5-1.5B); models with one key/value head per query head keep the default fused attention
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model); gqa = getattr(cfg, "num_key_value_heads", None) not in (None, getattr(cfg, "num_attention_heads", None))
    kw = dict(attn_implementation="eager") if (dev == "mps" and gqa) else {}
    print(f"attention: {'eager' if kw else 'default'}; samples per chunk: {a.chunk or K.KSAMP}", flush=True)
    try: m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, **kw)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, **kw)   # older transformers
    m = m.to(dev); m.eval()
    shared = [not a.no_share]
    t0 = time.time()
    with open(a.out, "a" if a.resume else "w") as fh:
        for n, p in enumerate(todo, 1):
            torch.manual_seed(K.SEED + p["id"])
            ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids): ids = ids["input_ids"]           # newer transformers return a BatchEncoding
            ids = ids.to(dev)
            g = None
            if shared[0]:
                try: g = gen_shared(m, tok, ids, torch, a.chunk)
                except Exception as e:
                    shared[0] = False; print(f"\nshared-prefix mode unavailable ({type(e).__name__}: {e}); using plain generate", flush=True)
            if g is None:
                with torch.no_grad():
                    g = m.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=5,
                                   num_return_sequences=K.KSAMP, pad_token_id=tok.eos_token_id)
            texts = [tok.decode(x[ids.shape[1]:], skip_special_tokens=True) for x in g]
            fh.write(json.dumps(dict(id=p["id"], samples=texts)) + "\n"); fh.flush()
            del g, ids                                   # return GPU memory each prompt, so the pool does not grow with prompt length
            if dev == "mps": torch.mps.empty_cache()
            elif dev == "cuda": torch.cuda.empty_cache()
            el = time.time() - t0
            print(f"\r{len(done) + n}/{len(prompts)}  {el / n:.1f}s per prompt  ETA {el / n * (len(todo) - n) / 60:.0f} min   ", end="", flush=True)
    print("\ndone", flush=True)

if __name__ == "__main__":
    main()
