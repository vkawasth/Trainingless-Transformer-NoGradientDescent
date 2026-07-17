"""
interaction_graph.py
====================
MEASURE THE BLOCK-TO-BLOCK GRADIENT COUPLING ON A REAL TRANSFORMER, across
training, and export it in the format comm_planner.py consumes.

This replaces the synthetic graph in comm_planner with a MEASURED one. It hooks
a real (small) transformer whose parameter names match your group_of() taxonomy
exactly, so gauge_probe.py / jacobian_rank.py / tangent_bundle.py run on it
unchanged, and so pointing this at YOUR model is a one-line swap of the
constructor.

WHAT "COUPLING" MEANS HERE, MEASURED NOT ASSERTED
-------------------------------------------------
The log's phase-1 claim was: "perturbing WQ/WK/WV/WO doesn't change FF's
gradients at all." That is exactly the cross-Hessian block

        H_{AB} = d^2 L / d theta_A d theta_B        ( = d grad_A / d theta_B )

If H_{attn,FF} ~ 0, attention and FF are gradient-decoupled: moving one does not
move the other's gradient. We estimate ||H_{AB}|| with Hessian-vector products:
for a random unit v supported on block B, (H v) restricted to A has RMS norm
proportional to ||H_{AB}||. No basis, no SVD, no ill-conditioning -- HVPs only.

The leading indicator r = ||g_att|| / ||g_ff|| is read off the same backward
pass for free.

OUTPUT
    measured_graph.json   (drop-in for comm_planner.load_measured_graph)
"""

from __future__ import annotations
import json
import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- taxonomy (identical to gauge_probe / jacobian_rank) ------------------ #
def group_of(name: str) -> str:
    n = name.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"):
        return "LayerNorm"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"):
        return "Emb"
    if ".ff." in n:
        return "FF"
    if "wk" in n:
        return "W_K"
    if "wq" in n:
        return "W_Q"
    if "wv" in n:
        return "W_V"
    if ".op." in n:
        return "W_O"
    return "other"


GROUPS = ["Emb", "W_Q", "W_K", "W_V", "W_O", "FF", "LayerNorm"]
ATTN = {"W_Q", "W_K", "W_V", "W_O"}


# ----- a non-memorizing Markov corpus (entropy floor = log k) --------------- #
class MarkovCorpus:
    """Order-1 chain: each state has exactly `succ` equally-likely successors,
    so the conditional entropy is log(succ) and no model can drive loss below
    it. succ=3 -> floor = ln 3 ~= 1.0986, matching the log."""
    def __init__(self, vocab=9, succ=3, seq=64, batch=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.vocab, self.seq, self.batch = vocab, seq, batch
        self.T = torch.zeros(vocab, vocab)
        for s in range(vocab):
            nxt = torch.randperm(vocab, generator=g)[:succ]
            self.T[s, nxt] = 1.0 / succ
        self.floor = math.log(succ)
        self.g = torch.Generator().manual_seed(seed + 1)

    def get_batch(self):
        B, L = self.batch, self.seq
        x = torch.zeros(B, L + 1, dtype=torch.long)
        x[:, 0] = torch.randint(self.vocab, (B,), generator=self.g)
        for t in range(L):
            probs = self.T[x[:, t]]
            x[:, t + 1] = torch.multinomial(probs, 1, generator=self.g).squeeze(1)
        return x[:, :-1], x[:, 1:]


class CopyCorpus:
    """Delayed-copy: y[t] = x[t-lag].  Inputs are random every batch, so the
    model cannot memorise -- it must learn the MECHANISM (attend lag positions
    back), which Emb/FF alone cannot do.  This is a task that PROVABLY needs
    attention, and is the positive control for 'does attention couple in?'."""
    def __init__(self, vocab=9, lag=3, seq=64, batch=32, seed=0):
        self.vocab, self.lag, self.seq, self.batch = vocab, lag, seq, batch
        self.floor = 0.0
        self.g = torch.Generator().manual_seed(seed + 1)

    def get_batch(self):
        B, L, lag = self.batch, self.seq, self.lag
        x = torch.randint(self.vocab, (B, L), generator=self.g)
        y = torch.zeros_like(x)
        y[:, lag:] = x[:, :-lag]
        y[:, :lag] = x[:, :lag]         # first `lag` positions: identity
        return x, y


# ----- a transformer whose param NAMES satisfy group_of() ------------------- #
class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)      # -> ".op." => W_O

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, L, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, L, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, L, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
        mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.op(o)


class FF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)             # -> ".ff." => FF
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(d), nn.LayerNorm(d)])  # ".ln." => LN
        self.attn = Attn(d, h)
        self.ff = FF(d)

    def forward(self, x):
        x = x + self.attn(self.ln[0](x))
        x = x + self.ff(self.ln[1](x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, seq, d=64, h=4, layers=3):
        super().__init__()
        self.te = nn.Embedding(vocab, d)           # "te" => Emb
        self.pe = nn.Embedding(seq, d)             # "pe" => Emb
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)                # "ln_f" => LayerNorm
        self.head = nn.Linear(d, vocab, bias=False)  # "head" => Emb

    def forward(self, idx, targets=None):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.te(idx) + self.pe(pos)[None]
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss


# ----- the measurement ------------------------------------------------------ #
def group_masks(model):
    """boolean index (over the flat param vector) for each group."""
    idx, masks = 0, {g: [] for g in GROUPS}
    slices = {}
    for name, p in model.named_parameters():
        g = group_of(name)
        n = p.numel()
        slices.setdefault(g, []).append((idx, idx + n))
        idx += n
    P = idx
    out = {}
    for g, segs in slices.items():
        if g not in GROUPS:
            continue
        m = torch.zeros(P, dtype=torch.bool)
        for a, b in segs:
            m[a:b] = True
        out[g] = m
    return out, P


def flat_grad(model, loss, create_graph=False):
    params = [p for _, p in model.named_parameters()]
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads]), params


def measure_graph(model, corpus, probes=4, seed=0):
    """Return (coupling[frozenset{A,B}] -> strength, r_indicator)."""
    x, y = corpus.get_batch()
    _, loss = model(x, y)
    g_full, params = flat_grad(model, loss, create_graph=True)
    masks, P = group_masks(model)

    # leading indicator: ||g_att|| / ||g_ff||
    g_att = torch.cat([g_full[masks[a]] for a in ATTN if a in masks]).norm().item()
    g_ff = g_full[masks["FF"]].norm().item() if "FF" in masks else 1.0
    r_ind = g_att / (g_ff + 1e-12)

    gen = torch.Generator().manual_seed(seed)
    # coupling[A][B] accumulates ||(Hv_B) restricted to A||
    present = [g for g in GROUPS if g in masks]
    acc = {a: {b: 0.0 for b in present} for a in present}
    for B in present:
        for _ in range(probes):
            v = torch.zeros(P)
            vb = torch.randn(int(masks[B].sum()), generator=gen)
            vb /= vb.norm() + 1e-12
            v[masks[B]] = vb
            # H v  via  grad( <g, v>, params )
            gv = (g_full * v).sum()
            Hv = torch.autograd.grad(gv, params, retain_graph=True)
            Hv = torch.cat([h.reshape(-1) for h in Hv])
            for A in present:
                acc[A][B] += float(Hv[masks[A]].norm())
    for A in present:
        for B in present:
            acc[A][B] /= probes

    coupling = {}
    for A, Bb in combinations(present, 2):
        s = 0.5 * (acc[A][Bb] + acc[Bb][A])        # symmetrise
        coupling[frozenset({A, Bb})] = s
    # self-coupling (diagonal) kept separately if useful
    diag = {A: acc[A][A] for A in present}
    return coupling, r_ind, diag


# ----- driver: train, measure at checkpoints, export ------------------------ #
def run_experiment(corpus, tag, checkpoints=(0, 25, 75, 150, 250, 400, 600)):
    torch.manual_seed(0)
    model = GPT(vocab=corpus.vocab, seq=corpus.seq, d=64, h=4, layers=3)
    P = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)

    print("=" * 78)
    print(f"  MEASURED INTERACTION GRAPH -- task: {tag}")
    print("=" * 78)
    print(f"  P = {P:,}   loss floor = {corpus.floor:.4f}")
    print(f"\n  {'step':>5}{'loss':>8}{'r=|g_att|/|g_ff|':>18}"
          f"{'Emb-FF':>9}{'attn-FF':>9}{'attn-attn':>11}")
    print("  " + "-" * 66)

    records = []
    step = 0
    for target in checkpoints:
        while step < target:
            x, y = corpus.get_batch()
            _, loss = model(x, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
        # measure
        x, y = corpus.get_batch()
        _, lv = model(x, y)
        coup, r_ind, diag = measure_graph(model, corpus, probes=4)

        def mean_between(groups_a, groups_b):
            vals = [w for e, w in coup.items()
                    if (any(g in groups_a for g in e) and
                        any(g in groups_b for g in e) and
                        not (set(e) <= groups_a and groups_a == groups_b and len(e) < 2))]
            return sum(vals) / len(vals) if vals else 0.0

        emb_ff = coup.get(frozenset({"Emb", "FF"}), 0.0)
        attn_ff = sum(coup.get(frozenset({a, "FF"}), 0.0) for a in ATTN) / 4
        attn_attn = sum(coup.get(frozenset({a, b}), 0.0)
                        for a, b in combinations(ATTN, 2)) / 6
        print(f"  {step:>5}{float(lv):>8.3f}{r_ind:>18.3f}"
              f"{emb_ff:>9.3f}{attn_ff:>9.3f}{attn_attn:>11.3f}")
        records.append({
            "step": step, "loss": float(lv), "r_indicator": r_ind,
            "coupling": {"|".join(sorted(e)): w for e, w in coup.items()},
            "self_coupling": diag,
        })

    allw = [w for rec in records for w in rec["coupling"].values()]
    hi = max(allw) if allw else 1.0
    for rec in records:
        rec["coupling_norm"] = {k: v / hi for k, v in rec["coupling"].items()}

    fname = f"measured_graph_{tag}.json"
    json.dump({"P": P, "task": tag, "loss_floor": corpus.floor,
               "groups": GROUPS, "checkpoints": records},
              open(fname, "w"), indent=2, default=float)
    print(f"\n  wrote {fname}")
    return records


def attn_ff_curve(records):
    return [(r["step"],
             sum(r["coupling"].get("|".join(sorted([a, "FF"])), 0.0)
                 for a in ATTN) / 4,
             r["coupling"].get("Emb|FF", r["coupling"].get("FF|Emb", 0.0)),
             r["r_indicator"]) for r in records]


def main():
    markov = run_experiment(MarkovCorpus(vocab=9, succ=3, seq=64, batch=32),
                            "markov_attn_not_needed")
    print()
    copy = run_experiment(CopyCorpus(vocab=9, lag=3, seq=64, batch=32),
                          "copy_attn_required")

    print("\n" + "=" * 78)
    print("  CONTRAST: does attention couple in when the task NEEDS it?")
    print("=" * 78)
    print(f"  {'task':<24}{'r early':>9}{'r late':>9}"
          f"{'attnFF early':>14}{'attnFF late':>13}   verdict")
    print("  " + "-" * 74)
    for tag, recs in [("markov (attn not needed)", markov),
                      ("copy   (attn required)", copy)]:
        c = attn_ff_curve(recs)
        r_e, r_l = c[0][3], c[-1][3]
        a_e, a_l = c[0][1], c[-1][1]
        rises = (r_l > r_e * 1.3) or (a_l > a_e * 1.3)
        v = "COUPLES IN" if rises else "decouples"
        print(f"  {tag:<24}{r_e:>9.3f}{r_l:>9.3f}{a_e:>14.4f}{a_l:>13.4f}   {v}")
    print("\n  If the two rows differ, the log's 'FF-first-then-attention'")
    print("  ordering is TASK-CONTINGENT, not a universal law of transformer")
    print("  training -- and the sparse-communication window your scheduler")
    print("  exploits exists only for tasks that actually recruit attention.")
    print("\n  Feed the matching measured_graph_*.json into comm_planner to")
    print("  price placements on the REAL graph.")


if __name__ == "__main__":
    main()
