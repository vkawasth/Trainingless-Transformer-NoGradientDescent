"""QUANTIZATION FRONTIER, with the eigsh determinism fix applied.

Every prior comparative sweep in this codebase called build() once per arm in a
single process, and spla.eigsh draws its Lanczos start vector from numpy's
global RNG. Measured here: three builds with identical seeds give models
differing by ||w0-w1|| = 43.2 and initial loss spread 0.029. Comparisons
coarser than that are safe; the 4-bit-vs-uncompressed gap (0.009) is not.

Fixed by pinning v0 and fixing eigenvector sign. ASSERTED, not assumed.

Arms: uncompressed, uniform quantization at b in {1,2,4,8} bits/coordinate,
stochastic rounding, per-tensor scale. Seeds vary the optimizer/batch stream;
model init is now identical across all arms by construction.
"""
import io, contextlib, subprocess, sys, math, json, time
import numpy as np, torch

subprocess.run([sys.executable, "build_corpus.py", "--out", "/tmp", "--loops", "300"],
               check=True, capture_output=True)

RAW = open("compiler_geometri_patched_86.py").read()
SRC = RAW[:RAW.find("# \u2500\u2500 PHASE 3")]
for old, new in [("D=256; N_HEADS=4", "D=128; N_HEADS=4"),
                 ("for mf_r in range(1, 16):", "for mf_r in range(1, 3):"),
                 ("    if pc == N_STU-1:", "    if False:"),
                 ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                  "    if False:")]:
    assert SRC.count(old) == 1, f"prelude anchor {old!r}: {SRC.count(old)}"
    SRC = SRC.replace(old, new, 1)

EIG_OLD = "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EIG_NEW = ("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
           "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
           "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EIG_OLD) == 1, f"eigsh anchor: {SRC.count(EIG_OLD)}"
SRC = SRC.replace(EIG_OLD, EIG_NEW, 1)
assert "v0=_v0" in SRC, "eigsh patch did not land"

def build():
    torch.manual_seed(1234); np.random.seed(1234)
    G = {}; b = io.StringIO()
    with contextlib.redirect_stdout(b): exec(SRC, G)
    return G

NS = 140

def quant(t, bits, gen):
    """Uniform per-tensor quantization, stochastic rounding.
    At bits=1 the grid is {min,max} which wastes both levels on outliers, so
    1-bit is reported but is not the best 1-bit scheme (sign*mean is)."""
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 1e-30: return t.clone()
    L = 2 ** bits - 1
    s = (hi - lo) / L
    q = (t - lo) / s
    fl = torch.floor(q)
    frac = q - fl
    r = torch.rand(t.shape, generator=gen)
    fl = fl + (r < frac).to(t.dtype)
    return fl.clamp(0, L) * s + lo

def signonly(t):
    fl = t.reshape(-1)
    return (torch.sign(fl) * float(fl.abs().mean())).view_as(t)

def run(mode, bits, seed):
    G = build(); model = G["model"]; get_batch = G["get_batch"]; LR = G["LR"]
    MATS = [(n, p) for n, p in model.named_parameters() if p.dim() == 2 and p.requires_grad]
    EV = [get_batch() for _ in range(8)]
    def L():
        t = 0.0
        with torch.no_grad():
            for x, y in EV: t += float(model(x, y)[1])
        return t / len(EV)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9, 0.95), weight_decay=0.1)
    rets = []; relerrs = []
    for st in range(NS):
        th = {n: p.data.clone() for n, p in MATS}
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        u = {n: (p.data - th[n]).clone() for n, p in MATS}
        if mode == "none": continue
        c = {n: (quant(u[n], bits, gen) if mode == "quant" else signonly(u[n]))
             for n, _ in MATS}
        un = math.sqrt(sum(float((u[n]*u[n]).sum()) for n, _ in MATS))
        cn = math.sqrt(sum(float((c[n]*c[n]).sum()) for n, _ in MATS))
        rets.append(cn / max(un, 1e-30))
        relerrs.append(math.sqrt(sum(float(((c[n]-u[n])**2).sum()) for n, _ in MATS)) / max(un, 1e-30))
        sc = un / cn if cn > 1e-30 else 1.0
        with torch.no_grad():
            for n, p in MATS: p.data.copy_(th[n] + c[n] * sc)
    v = L()
    del G, model
    import gc; gc.collect()
    return dict(final=v, ret=float(np.mean(rets)) if rets else 1.0,
                relerr=float(np.mean(relerrs)) if relerrs else 0.0)

if __name__ == "__main__":
    # determinism check FIRST
    w = []
    for _ in range(2):
        G = build()
        w.append(torch.cat([p.data.reshape(-1) for p in G["model"].parameters()]).clone())
        del G
    d = float((w[0] - w[1]).norm())
    print(f"  determinism check: ||w0-w1|| = {d:.3e}   (was 43.2 unpatched)")
    assert d < 1e-6, "eigsh fix did not take"
    print()
    SEEDS = [17, 18, 19]
    arms = [("none", 0), ("quant", 8), ("quant", 4), ("quant", 2), ("quant", 1), ("signonly", 1)]
    res = {}
    for mode, b in arms:
        for s in SEEDS:
            t0 = time.time()
            res[(mode, b, s)] = run(mode, b, s)
            print(f"  {mode} b={b} seed={s}: {res[(mode,b,s)]['final']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  {'arm':>12}{'mean':>9}{'sd':>8}{'vs none':>9}{'relerr':>9}")
    base = np.mean([res[("none", 0, s)]["final"] for s in SEEDS])
    for mode, b in arms:
        f = [res[(mode, b, s)]["final"] for s in SEEDS]
        e = np.mean([res[(mode, b, s)]["relerr"] for s in SEEDS])
        nm = "uncompressed" if mode == "none" else (f"quant {b}bit" if mode == "quant" else "signonly")
        print(f"  {nm:>12}{np.mean(f):>9.4f}{np.std(f):>8.4f}{np.mean(f)/base:>9.3f}{e:>9.3f}")
    json.dump({f"{m}_{b}_{s}": res[(m, b, s)] for m, b in arms for s in SEEDS},
              open("res_quant2.json", "w"), indent=1)
