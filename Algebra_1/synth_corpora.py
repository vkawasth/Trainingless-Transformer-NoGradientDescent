"""
Synthetic corpus suite with GROUND-TRUTH structure.

Grammar (byte-level, so it drops straight into the V=256 harness):

    S -> E ;
    E -> T | E+T | E-T
    T -> F | T*F
    F -> V | D | ( E )
    V -> a..h        D -> 0..9

Everything is generated from an explicit parse tree, so both the symmetries and
the structural distances are exact by construction rather than inferred.

EXACT SYMMETRIES (parse tree preserved or mirrored; surface form changes):
  ws      : whitespace around binary operators. Parse tree IDENTICAL.
            The purest surface-only transform available -- zero structural change.
  rename  : consistent bijection on the variable alphabet (alpha-renaming).
            Tree shape identical, terminals relabelled.
  comm    : swap operands of the commutative ops (+, *). Value-preserving,
            tree mirrored at those nodes.

STRUCTURAL LADDER (known distance, for strand-vs-structure regression):
  0 identity | 1 one terminal | 2 one operator (production)
  3 one subtree | 4 full resample

CONTROL:
  shuf    : token-shuffled. Same unigram stats, zero structure.
  lexctl  : same token histogram as `rename` but structure destroyed -- separates
            "invariant because equivalent" from "invariant because same letters".

Live vocab is ~26 bytes, deliberately close to english (27), so cfg sits on the
existing structure axis as a vocab-matched comparison rather than a new regime.
"""
import json, os, gzip, math, numpy as np
from collections import Counter

V = 256
NTOK = 60000
OUT = "/home/claude/work/bundle/corpora"
VARS = "abcdefgh"
DIGS = "0123456789"
rng = np.random.default_rng(11)


# ---------- tree construction ----------
def gen_E(d=0):
    if d > 3 or rng.random() < 0.35:
        return gen_T(d + 1)
    return ("bin", "+-"[rng.integers(2)], gen_E(d + 1), gen_T(d + 1))


def gen_T(d=0):
    if d > 3 or rng.random() < 0.5:
        return gen_F(d + 1)
    return ("bin", "*", gen_T(d + 1), gen_F(d + 1))


def gen_F(d=0):
    p = rng.random()
    if p < 0.45:
        return ("var", VARS[rng.integers(len(VARS))])
    if p < 0.85 or d > 3:
        return ("dig", DIGS[rng.integers(10)])
    return ("par", gen_E(d + 1))


COMMUTATIVE = {"+", "*"}


def render(t, ws=False, rename=None, comm=False):
    k = t[0]
    if k == "var":
        c = t[1]
        return (rename or {}).get(c, c)
    if k == "dig":
        return t[1]
    if k == "par":
        return "(" + render(t[1], ws, rename, comm) + ")"
    _, op, a, b = t
    if comm and op in COMMUTATIVE:
        a, b = b, a
    sep = " " + op + " " if ws else op
    return render(a, ws, rename, comm) + sep + render(b, ws, rename, comm)


# ---------- structural perturbations ----------
def nodes(t, path=()):
    yield path, t
    if t[0] == "par":
        yield from nodes(t[1], path + (1,))
    elif t[0] == "bin":
        yield from nodes(t[2], path + (2,))
        yield from nodes(t[3], path + (3,))


def replace(t, path, new):
    if not path:
        return new
    i, rest = path[0], path[1:]
    l = list(t)
    l[i] = replace(t[i], rest, new)
    return tuple(l)


def perturb(t, level):
    """level: 1 terminal, 2 operator, 3 subtree, 4 full resample."""
    if level == 4:
        return gen_E()
    ns = list(nodes(t))
    if level == 1:
        cand = [p for p, n in ns if n[0] in ("var", "dig")]
        if not cand:
            return t
        p = cand[rng.integers(len(cand))]
        old = dict(ns)[p]
        if old[0] == "dig":
            new = ("dig", DIGS[rng.integers(10)])
        else:
            new = ("var", VARS[rng.integers(len(VARS))])
        return replace(t, p, new)
    if level == 2:
        cand = [p for p, n in ns if n[0] == "bin"]
        if not cand:
            return t
        p = cand[rng.integers(len(cand))]
        old = dict(ns)[p]
        ops = [o for o in "+-*" if o != old[1]]
        return replace(t, p, ("bin", ops[rng.integers(len(ops))], old[2], old[3]))
    if level == 3:
        cand = [p for p, n in ns if p]
        if not cand:
            return gen_E()
        p = cand[rng.integers(len(cand))]
        return replace(t, p, gen_F())
    return t


# ---------- build ----------
def to_bytes(s):
    return list(s.encode("ascii"))


def stream(exprs, **kw):
    out = []
    for t in exprs:
        out += to_bytes(render(t, **kw) + ";")
    return out


def pad(a, n=NTOK):
    a = list(a)
    if len(a) < n:
        a = (a * (n // len(a) + 1))[:n]
    return a[:n]


def gzip_ratio(a):
    b = np.asarray(a, dtype=np.uint8).tobytes()
    return len(gzip.compress(b, 9)) / len(b)


def bent(x, L=6):
    blk = [tuple(x[i:i + L]) for i in range(0, len(x) - L, 2)]
    c = Counter(blk); n = len(blk)
    return -sum((v / n) * math.log2(v / n) for v in c.values()) / L


def gap(a, L=6):
    a = np.asarray(a); sh = a.copy(); rng.shuffle(sh)
    return bent(sh, L) - bent(a, L)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # base derivations, enough to overfill NTOK
    exprs = [gen_E() for _ in range(6000)]
    ren = {c: VARS[(i + 3) % len(VARS)] for i, c in enumerate(VARS)}

    variants = {
        "cfg":        dict(),
        "cfg_ws":     dict(ws=True),
        "cfg_rename": dict(rename=ren),
        "cfg_comm":   dict(comm=True),
    }
    built = {}
    for name, kw in variants.items():
        built[name] = pad(stream(exprs, **kw))

    # controls
    a = np.array(built["cfg"]); rng.shuffle(a); built["cfg_shuf"] = a.tolist()
    a = np.array(built["cfg_rename"]); rng.shuffle(a); built["cfg_lexctl"] = a.tolist()

    for name, s in built.items():
        json.dump(s, open(f"{OUT}/{name}.json", "w"))

    # aligned expression lists for the paired-batch experiment
    keep = 4000
    pairs = {
        "base":   [render(t) for t in exprs[:keep]],
        "ws":     [render(t, ws=True) for t in exprs[:keep]],
        "rename": [render(t, rename=ren) for t in exprs[:keep]],
        "comm":   [render(t, comm=True) for t in exprs[:keep]],
    }
    for lv in (1, 2, 3, 4):
        pairs[f"pert{lv}"] = [render(perturb(t, lv)) for t in exprs[:keep]]
    json.dump(pairs, open(f"{OUT}/cfg_pairs.json", "w"))

    print(f"{'corpus':>14}{'vocab':>7}{'gzip':>9}{'blockH-gap':>12}")
    for name, s in built.items():
        print(f"{name:>14}{len(set(s)):>7}{gzip_ratio(s):>9.3f}{gap(s):>12.3f}")

    # how much surface actually changed under each exact symmetry
    print("\n  surface change under exact symmetries (byte-identity vs base):")
    b = pairs["base"]
    for k in ("ws", "rename", "comm"):
        same = np.mean([x == y for x, y in zip(b, pairs[k])])
        print(f"    {k:>7}: {100*(1-same):5.1f}% of expressions altered")
    print("\n  mean expression length:", round(np.mean([len(x) for x in b]), 1))
