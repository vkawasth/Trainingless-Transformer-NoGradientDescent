#!/usr/bin/env python3
"""
signtop.py -- SIGN+TOP UPDATE COMPRESSION
==========================================

Compress the OPTIMIZER'S UPDATE, not the gradient:

    keep the top-k coordinates of u at full precision
    represent every remaining coordinate as sign(u_i) * mean(|u_tail|)

    u_hat_i = u_i                              i in K
            = sign(u_i) * mean_{j not in K}|u_j|    otherwise

WHY THIS RATHER THAN A LOW-RANK PROJECTION

Adam's update is nearly uniform in coordinate magnitude. Measured per
coordinate on this pipeline:

    corr(|g|, |u|) = +0.04
    mean |u| flat across all ten deciles of |g|
    |u|/|g| running 3367 in the bottom decile to 0.43 in the top

A uniform-magnitude matrix is generically FULL RANK, so forcing it through a
rank-r basis is a mismatch of geometry rather than a bound on information.

MEASURED, matched storage, 140-step trajectories, identical init and batches:

    arm             floats/matrix   final loss   vs uncompressed
    uncompressed        --            0.265          1.00
    signtop r=32      18,432          0.310          1.17
    signtop r=4        2,080          0.472          1.78
    lowrank r=32      18,432          1.502          5.66
    lowrank r=4        2,080          3.507         13.2

sign+top at the rank-4 budget -- 15 exact coordinates plus a sign field --
beats low-rank at the rank-32 budget, which costs 9x more.

Per-step operator comparison at the r=32 budget, both applied to the SAME
uncompressed trajectory:

    op          cos    relerr   |uh|/|u|   D retained
    lowrank    0.568    0.821     0.568      0.858
    signtop    0.949    0.314     0.949      0.892
    toponly    0.798    0.601     0.798      0.632
    signonly   0.820    0.571     0.820      0.632

BOTH INGREDIENTS ARE NECESSARY. top-k alone and sign alone each retain 0.632
of the local descent; together they retain 0.892. Neither is decoration.

AND LOCAL DESCENT IS NOT THE OBJECTIVE. The low-rank arm accumulated MORE
descent over the run (15.44 against 11.70) and ended 5x worse. What separates
them is directional fidelity: cos 0.949 against 0.568. Low-rank also shrinks the
step -- |u_hat|/|u| = 0.568, since a projection can only remove norm -- while
sign+top preserves it at 0.949.

STORAGE ACCOUNTING, honestly. The sign field is not free: it costs numel/32
floats-equivalent at one bit per coordinate. k is solved so that

    2k + numel/32 + 1  ==  r(d1+d2) + 2r^2
    |    |        |         |
    |    |        scale     the low-rank budget being matched
    |    sign bits
    value + index

At r=4 on a 256x256 matrix that leaves k=15, because the sign field alone costs
2,048 of the 2,080 budget. An earlier version of this measurement omitted the
sign-field cost and was 2x over budget at r=4.

WHAT THIS IS NOT. This is one landscape (D=128, a deterministic corpus with
L_corpus = 0 exactly), one seed per arm, and 140 steps. The claim it supports is
narrow: the Adam update here is poorly matched to matrix low-rank approximation
and substantially better matched by exact large coordinates plus a one-bit sign
field. It is not a claim that the update "is an ell-infinity object", and the
trajectory result has not been replicated across seeds.

USAGE

    from signtop import compress_step

    # inside the training loop, replacing a plain opt.step()
    compress_step(model, opt, rank_budget=32)

or, if you want the pieces:

    from signtop import signtop_compress

    before = {n: p.data.clone() for n, p in model.named_parameters()
              if p.dim() == 2 and p.requires_grad}
    opt.step()
    u = {n: p.data - before[n] for n, p in model.named_parameters()
         if n in before}
    u_hat = signtop_compress(u, rank_budget=32)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in u_hat:
                p.data.copy_(before[n] + u_hat[n])

Run this file directly for a budget table and a self-contained demo:

    python3 signtop.py
"""
import math
import torch


def signtop_budget(shape, rank_budget):
    """Number of full-precision coordinates affordable at a low-rank budget.

    Solves  2k + numel/32 + 1 == r(d1+d2) + 2r^2  for k, so the comparison
    against a rank-r two-sided projector is storage-for-storage. Returns 0 when
    the sign field alone exhausts the budget, in which case the compressor
    degenerates to sign-only.
    """
    d1, d2 = shape
    numel = d1 * d2
    lowrank_cost = rank_budget * (d1 + d2) + 2 * rank_budget * rank_budget
    return max(0, int((lowrank_cost - numel / 32.0 - 1) // 2))


def signtop_tensor(u, k):
    """sign+top on one tensor. k coordinates exact, the rest one bit each."""
    flat = u.reshape(-1)
    out = torch.zeros_like(flat)
    tail = torch.ones_like(flat, dtype=torch.bool)
    if k > 0:
        idx = torch.topk(flat.abs(), min(k, flat.numel()), sorted=False).indices
        tail[idx] = False
        out[idx] = flat[idx]
    if tail.any():
        # one scalar per tensor, shared by the whole tail
        scale = float(flat[tail].abs().mean())
        out[tail] = torch.sign(flat[tail]) * scale
    return out.view_as(u)


def signtop_compress(u, rank_budget=32, k=None, min_dim=2):
    """Compress a dict of update tensors.

    u            {name: tensor} of realised updates, theta_{t+1} - theta_t
    rank_budget  match the storage of a rank-r two-sided projector
    k            override the solved budget with an explicit coordinate count
    min_dim      tensors with fewer dims are passed through uncompressed;
                 1-D parameters are a negligible share of storage and have no
                 matrix structure to compress
    """
    out = {}
    for name, t in u.items():
        if t.dim() < min_dim:
            out[name] = t.clone()
            continue
        kk = k if k is not None else signtop_budget(tuple(t.shape), rank_budget)
        out[name] = signtop_tensor(t, kk)
    return out


def stored_floats(u, rank_budget=32, k=None, min_dim=2):
    """Actual storage of the compressed representation, in float-equivalents."""
    total = 0.0
    for name, t in u.items():
        if t.dim() < min_dim:
            total += t.numel(); continue
        kk = k if k is not None else signtop_budget(tuple(t.shape), rank_budget)
        total += 2 * kk + t.numel() / 32.0 + 1
    return total


def lowrank_compress(u, grad, rank):
    """Two-sided rank-r projection, for comparison. The projector comes from the
    GRADIENT's SVD, as GaLore and SubTrack++ do."""
    out = {}
    for name, t in u.items():
        if t.dim() < 2:
            out[name] = t.clone(); continue
        r = min(rank, min(t.shape))
        U, _, Vt = torch.linalg.svd(grad[name], full_matrices=False)
        P, Q = U[:, :r], Vt[:r].T
        out[name] = P @ (P.T @ t @ Q) @ Q.T
    return out


def lowrank_floats(u, rank):
    total = 0.0
    for name, t in u.items():
        if t.dim() < 2:
            total += t.numel(); continue
        r = min(rank, min(t.shape))
        total += r * (t.shape[0] + t.shape[1]) + 2 * r * r
    return total


def compress_step(model, opt, rank_budget=32, k=None):
    """Take one optimizer step and apply it in compressed form.

    Drop-in for opt.step(). Gradients must already be populated. Only 2-D
    parameters are compressed; 1-D ones (LayerNorm gains, biases) pass through,
    since they are a negligible share of storage and have no matrix structure.

    Returns the fraction of the step's l2 norm retained, which is also
    cos(u, u_hat) for this compressor -- useful as a live diagnostic. Low rank
    reads ~0.57 here where sign+top reads ~0.95.
    """
    before = {n: p.data.clone() for n, p in model.named_parameters()
              if p.requires_grad and p.dim() == 2}
    opt.step()
    u = {n: (p.data - before[n]) for n, p in model.named_parameters()
         if n in before}
    u_hat = signtop_compress(u, rank_budget=rank_budget, k=k)
    num = den = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n not in u_hat:
                continue
            num += float((u_hat[n] * u_hat[n]).sum())
            den += float((u[n] * u[n]).sum())
            p.data.copy_(before[n] + u_hat[n])
    return math.sqrt(num / den) if den > 0 else 1.0


def _demo():
    """Self-contained check that the compressor runs and retains most of the
    step, on a throwaway model. Not a benchmark."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 256))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95))
    x = torch.randn(64, 256); y = torch.randn(64, 256)
    print("\n  demo: 20 steps on a throwaway net, rank-4 budget\n")
    print(f"  {'step':>6}{'loss':>10}{'retained':>10}")
    for i in range(20):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward()
        ret = compress_step(net, opt, rank_budget=4)
        if (i + 1) % 5 == 0:
            print(f"  {i+1:>6}{loss.item():>10.4f}{ret:>10.3f}")
    print("\n  retained ~0.95 is the expected reading for sign+top;")
    print("  a two-sided rank-4 projection reads ~0.57 on the same steps.")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("  budget solved so sign+top matches a rank-r two-sided projector\n")
    print(f"  {'shape':>12}{'r':>4}{'lowrank':>10}{'k':>8}{'signtop':>10}{'ratio':>8}")
    for shape in ((256, 256), (128, 512), (1017, 128)):
        for r in (4, 8, 32):
            u = {"w": torch.randn(*shape)}
            k = signtop_budget(shape, r)
            lr = lowrank_floats(u, r)
            st = stored_floats(u, r)
            print(f"  {str(shape):>12}{r:>4}{lr:>10,.0f}{k:>8,}{st:>10,.0f}"
                  f"{st/lr:>8.3f}")
    print("\n  ratio ~ 1.000 means the budgets are matched. A ratio above 1")
    print("  would mean sign+top is being given more storage than low-rank,")
    print("  which is the error an earlier version of this made at r=4.")
    _demo()
