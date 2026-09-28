"""Causal transformer on Corpus B, with and without guidance from the fitted (split-merge) grammar.

variants
  base   plain next-token LM
  aux    + auxiliary heads trained to predict the fitted grammar's prefix posteriors
           P(B_l(t) | x_<t), l = 1..4 (distillation; nothing extra at test time)
  feed   + those posteriors (log) added to the input embedding at position t
           (the grammar is used at test time)
"""
import json, sys, time, argparse
import numpy as np, jax, jax.numpy as jnp, optax

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="base")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--aux-w", type=float, default=1.0)
ap.add_argument("--n-train", type=int, default=19000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--feats", default="cH/feats_fit.npz")
ap.add_argument("--out", default="")
a = ap.parse_args()

F = np.load(a.feats)
S, A, L, H = 16, 12, 4, 4
Xtr, Xdv, Xva = F["Xtr"][:a.n_train], F["Xdv"], F["Xva"]
anc = {k: [F[f"{k}_anc{l}"] for l in range(1, L + 1)] for k in ("tr", "dv", "va")}
anc["tr"] = [x[:a.n_train] for x in anc["tr"]]
Vl = [x.shape[-1] for x in anc["va"]]
feat = {k: np.concatenate([np.log(np.clip(x, 1e-6, None)) for x in anc[k]], -1).astype(np.float32) for k in anc}


def init(key):
    d = a.d; ks = iter(jax.random.split(key, 64)); n = lambda *s: jax.random.normal(next(ks), s) * 0.02
    p = {"emb": n(A + 1, d), "pos": n(S, d), "lnf": (jnp.ones(d), jnp.zeros(d)), "out": n(d, A),
         "blocks": [{"ln1": (jnp.ones(d), jnp.zeros(d)), "qkv": n(d, 3 * d), "o": n(d, d),
                     "ln2": (jnp.ones(d), jnp.zeros(d)), "w1": n(d, 4 * d), "w2": n(4 * d, d)}
                    for _ in range(a.layers)]}
    if a.variant == "feed": p["fin"] = n(sum(Vl), d)
    if a.variant == "aux": p["aux"] = [n(d, v) for v in Vl]
    return p


def ln(x, g):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g[0] + g[1]


mask = jnp.tril(jnp.ones((S, S), bool))


def forward(p, x, f):
    inp = jnp.concatenate([jnp.full((x.shape[0], 1), A), x[:, :-1]], 1)     # BOS shift
    h = p["emb"][inp] + p["pos"]
    if a.variant == "feed": h = h + f @ p["fin"]
    d = a.d; dh = d // H
    for b in p["blocks"]:
        q, k, v = jnp.split(ln(h, b["ln1"]) @ b["qkv"], 3, -1)
        sp = lambda z: z.reshape(z.shape[0], S, H, dh).transpose(0, 2, 1, 3)
        q, k, v = sp(q), sp(k), sp(v)
        att = jnp.where(mask, q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh), -1e9)
        z = (jax.nn.softmax(att, -1) @ v).transpose(0, 2, 1, 3).reshape(h.shape)
        h = h + z @ b["o"]
        h = h + jax.nn.gelu(ln(h, b["ln2"]) @ b["w1"]) @ b["w2"]
    h = ln(h, p["lnf"])
    return h @ p["out"], h


def nll_tok(p, x, f):
    lg, _ = forward(p, x, f)
    return -jnp.take_along_axis(jax.nn.log_softmax(lg), x[..., None], -1)[..., 0]


def loss(p, x, f, tg):
    lg, h = forward(p, x, f)
    l = -jnp.take_along_axis(jax.nn.log_softmax(lg), x[..., None], -1).mean()
    if a.variant == "aux":
        for W, t in zip(p["aux"], tg):
            l = l + a.aux_w * (-(t * jax.nn.log_softmax(h @ W)).sum(-1).mean()) / L
    return l


sched = optax.warmup_cosine_decay_schedule(0, a.lr, 200, a.steps)
opt = optax.adamw(sched, weight_decay=0.01)
p = init(jax.random.PRNGKey(a.seed)); st = opt.init(p)


@jax.jit
def step(p, st, x, f, tg):
    l, g = jax.value_and_grad(loss)(p, x, f, tg)
    u, st = opt.update(g, st, p)
    return optax.apply_updates(p, u), st, l


evalf = jax.jit(nll_tok)
rng = np.random.default_rng(a.seed)
best, bestp, t0 = 1e9, p, time.time()
for it in range(1, a.steps + 1):
    idx = rng.integers(0, len(Xtr), a.bs)
    p, st, l = step(p, st, jnp.array(Xtr[idx]), jnp.array(feat["tr"][idx]),
                    [jnp.array(x[idx], jnp.float32) for x in anc["tr"]])
    if it % 250 == 0 or it == a.steps:
        dv = float(np.asarray(evalf(p, jnp.array(Xdv), jnp.array(feat["dv"]))).sum(1).mean())
        if dv < best: best, bestp = dv, p
print(f"{a.variant} L{a.layers} seed{a.seed}: best dev {best:.4f}  ({time.time()-t0:.0f}s)", file=sys.stderr)
nv = np.asarray(evalf(bestp, jnp.array(Xva), jnp.array(feat["va"])))
if a.out: np.save(a.out, nv)
print(json.dumps({"variant": a.variant, "layers": a.layers, "seed": a.seed, "n_train": a.n_train,
                  "val_nll_seq": float(nv.sum(1).mean())}))
