#!/usr/bin/env python3
"""ATTENTION BLOCK REDUNDANCY DETECTOR

    python3 attn_pruner.py                       # run on the geometry compiler
    python3 attn_pruner.py --steps 400           # longer
    python3 attn_pruner.py --ablate 0            # verify a flagged block is inert

Needs compiler_geometri_patched_86.py and build_corpus.py (or build_corpus2.py)
in the same directory.

WHAT IT DETECTS
---------------
An attention block whose map P has collapsed to the causal-uniform
distribution is a static prefix average. The residual stream already carries
its input forward, so the loss is insensitive to P, dL/dP ~ 0, and the block's
W_Q, W_K receive no gradient. Adam's 1/sqrt(v) then manufactures a full-sized
step out of that null signal.

THE ANALYTIC FLOOR
------------------
With a causal mask, row i of P attends to i positions. If the logits carry no
information the softmax returns the uniform distribution over those i, so

    p_max(i) = 1/i        and       H(row i) = ln(i).

Averaging over the sequence gives two constants that depend on N alone:

    p_max  =  (1/N) sum_{i=1..N} 1/i     =  H_N / N
    ent    =  (1/N) sum_{i=1..N} ln(i)   =  ln(N!) / N
    (normalised by ln N: ln(N!)/(N ln N))

At N=64 these are 0.0741 and 0.7708. Measured on a frozen block 0 they are
0.0741 and 0.7708 -- four decimals, no fitting.

WHY THE GRADIENT VANISHES
-------------------------
With E = dL/d(PV),
    dL/dP = E V^T,
    g_Q   = X^T [ (1/sqrt(d_k)) J_softmax : (dL/dP) ] K,   J = diag(P) - P P^T.
If PV is a static prefix average of X and the residual already carries X, then
L is invariant to P, so dL/dP -> 0 and g_Q -> 0 regardless of J. This is why
the Jacobian magnitude does NOT explain the collapse: block 0's ||J||_F is
within 11% of block 1's while its gradient ratio is 310x smaller.

THE TWO-PART TEST
-----------------
Flag a block only if BOTH hold:

    m_p = (p_max - H_N/N) / (H_N/N)              < 0.10
    m_r = (||g_Q||/||g_V||) / (||J||_F/sqrt(d_k)) < 1.0

The first says the map is at the causal-uniform floor; the second says the
gradient is starved beyond what the softmax Jacobian accounts for. Requiring
both prevents a merely low-entropy block from being pruned.

MEASURED SEPARATION (D=256, 6 blocks, N=64)
    block 0:      m_p = 0.003,  m_r = 0.02      PRUNE
    blocks 1-5:   m_p = 0.47-1.16, m_r = 6.0-8.0  keep
Three orders of magnitude on m_r, two on m_p. The thresholds are not tuned;
anything in m_p 0.05-0.4 and m_r 0.1-3 gives the same answer.

VERIFIED CONSEQUENCE
    baseline  val@400 = 3.0531
    zero block 0 attention output   3.0492
    freeze W_Q, W_K at init         3.0528
    random (Xavier) init instead of spectral: still at the floor, 0.0801
Three independent ablations agree within 0.004, and random initialisation does
not rescue selectivity -- the redundancy is structural, not an artefact of the
initialiser.

NOT ESTABLISHED
    One corpus, one seed, 6 blocks, SEQ=64, D=256. Whether first blocks are
    routinely causal-uniform at greater depth or longer context is untested;
    the detector is cheap enough to simply run there.
"""
import io, contextlib, math, argparse, sys, os
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=240)
ap.add_argument("--probe-at", type=int, nargs="*", default=[40, 120, 240])
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--mp-thresh", type=float, default=0.10)
ap.add_argument("--mr-thresh", type=float, default=1.0)
ap.add_argument("--ablate", type=int, default=None,
                help="zero this block's attention output and report the cost")
ap.add_argument("--compiler", default="compiler_geometri_patched_86.py")
a = ap.parse_args()

if not os.path.exists("/tmp/train_ids.json"):
    sys.exit("no corpus in /tmp -- run build_corpus2.py --out /tmp first")

R = open(a.compiler).read()
SRC = R[:R.find("# \u2500\u2500 PHASE 3")]
for o, n in [("for mf_r in range(1, 16):", "for mf_r in range(1, 3):"),
             ("    if pc == N_STU-1:", "    if False:"),
             ("ETA_MF=0.01", "ETA_MF=0.002"),
             ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
              "    if False:")]:
    if SRC.count(o) != 1:
        sys.exit(f"anchor not found or not unique: {o!r}")
    SRC = SRC.replace(o, n, 1)
EO = "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN = ("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
      "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
      "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
SRC = SRC.replace(EO, EN, 1)

torch.manual_seed(1234); np.random.seed(1234)
G = {}; _b = io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(SRC, G)
model = G["model"]; gb = G["get_batch"]; LR = G["LR"] * 5
H = a.heads
if a.ablate is not None:
    with torch.no_grad():
        model.blocks[a.ablate].attn.op.weight.zero_()
    model.blocks[a.ablate].attn.op.weight.requires_grad_(False)
    print(f"  ABLATION: block {a.ablate} attention output zeroed\n")
P_ = {n: p for n, p in model.named_parameters() if p.requires_grad}
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
EV = [gb() for _ in range(4)]

def vloss():
    with torch.no_grad():
        return sum(float(model(x, y)[1]) for x, y in EV) / len(EV)

def probe():
    """per block: p_max, normalised entropy, predicted attenuation"""
    x, _ = gb(); out = []
    with torch.no_grad():
        h = model.te(x) + model.pe(torch.arange(x.shape[1]))
        for bm in model.blocks:
            B, T, Dm = h.shape; dh = Dm // H
            q = bm.attn.WQ(h).view(B, T, H, dh).transpose(1, 2)
            k = bm.attn.WK(h).view(B, T, H, dh).transpose(1, 2)
            s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
            s = s.masked_fill(torch.triu(torch.ones(T, T), 1).bool(), float("-inf"))
            P = s.softmax(-1).clamp_min(1e-12)
            s2 = (P ** 2).sum(-1); s3 = (P ** 3).sum(-1)
            jf = float(torch.sqrt((s2 - 2 * s3 + s2 ** 2).clamp_min(0)).mean())
            out.append((float(P.max(-1).values.mean()),
                        float((-(P * P.log()).sum(-1)).mean()) / math.log(T),
                        jf / math.sqrt(dh), T))
            h = bm(h)
    return out

flagged = {}
for t in range(1, a.steps + 1):
    x, y = gb(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                    if p.requires_grad], 1.0)
    if t in a.probe_at:
        pr = probe()
        N = pr[0][3]
        floor_p = sum(1.0 / i for i in range(1, N + 1)) / N
        floor_e = math.lgamma(N + 1) / (N * math.log(N))
        print(f"\n  t={t}   val={vloss():.4f}   N={N}   "
              f"floor p_max={floor_p:.4f}  ent={floor_e:.4f}")
        print(f"  {'blk':>4}{'p_max':>9}{'ent':>8}{'m_p':>8}"
              f"{'gQ/gV':>10}{'pred':>9}{'m_r':>8}   decision")
        for bi, (pm, en, pred, _) in enumerate(pr):
            gq = P_.get(f"blocks.{bi}.attn.WQ.weight")
            gv = P_.get(f"blocks.{bi}.attn.WV.weight")
            gqn = float(gq.grad.norm()) if gq is not None and gq.grad is not None else 0.0
            gvn = float(gv.grad.norm()) if gv is not None and gv.grad is not None else 1.0
            r = gqn / max(gvn, 1e-30)
            mp = (pm - floor_p) / floor_p
            mr = r / max(pred, 1e-30)
            hit = mp < a.mp_thresh and mr < a.mr_thresh
            flagged[bi] = flagged.get(bi, 0) + (1 if hit else 0)
            print(f"  {bi:>4}{pm:>9.4f}{en:>8.4f}{mp:>8.3f}{r:>10.2e}"
                  f"{pred:>9.4f}{mr:>8.2f}   {'PRUNE' if hit else 'keep'}")
    opt.step()

print(f"\n  final val {vloss():.4f}")
nb = len(model.blocks)
allhit = [b for b in range(nb) if flagged.get(b, 0) == len(a.probe_at)]
if allhit:
    d = model.te.weight.shape[1]
    print(f"\n  flagged at every probe: blocks {allhit}")
    print(f"  removable per block: W_Q + W_K = {2*d*d:,} params, their m and v,")
    print(f"  and one QK^T matmul + softmax per forward and backward.")
    print(f"  verify with:  python3 {sys.argv[0]} --ablate {allhit[0]}")
else:
    print(f"\n  no block flagged at every probe")
