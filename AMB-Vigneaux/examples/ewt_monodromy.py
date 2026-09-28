"""Arity-graded monodromy on Universal Dependencies English-EWT: an annotation-consistency audit.

Layers ("sources") on every word: X = XPOS (Penn tags) and B = UPOS + morphological FEATS.
X and B are nearly functions of each other, so a round trip X -> B -> X is the identity when every
genre follows the same conventions.  Provenance: the five genres (email, reviews, answers,
newsgroup, weblog).  For an ORDERED pair of genres (g1, g2) the loop is
      M = T_{X->B}^{g1}  T_{B->X}^{g2},       T[x, b] = P(b | x) estimated on genre g1, etc.
-- a 2-cycle in the provenance cover (the same pair of layers under two provenances, estimated on
disjoint sentences).  Grade = number of right dependents k of the word (and its parity).
A tag x is MOVED if the round trip sends most of its mass elsewhere: argmax_x' M[x, x'] != x with a
margin of 0.1.  Null: genre labels shuffled across sentences.  Positive control: in 'email', words
with ODD k tagged NN are relabelled NNP (a planted convention difference).

usage: python examples/ewt_monodromy.py /path/to/UD_English-EWT
"""
import sys, glob, collections, json, numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/UD_English-EWT"
MINC, MARGIN, NNULL = 30, 0.1, 20


def load():
    sents = []
    for f in sorted(glob.glob(f"{root}/*.conllu")):
        cur, g = [], None
        for line in open(f, encoding="utf8"):
            if line.startswith("# sent_id"):
                g = line.split("=")[1].strip().split("-")[0]
            elif not line.strip():
                if cur:
                    kids = collections.defaultdict(list)
                    for c in cur:
                        kids[c[6]].append(int(c[0]))
                    toks = [(c[4], c[3] + "|" + c[5], sum(k > int(c[0]) for k in kids.get(c[0], [])), c[1].lower()) for c in cur]
                    sents.append((g, toks))
                cur = []
            elif not line.startswith("#"):
                c = line.rstrip("\n").split("\t")
                if c[0].isdigit():
                    cur.append(c)
    return sents


SENTS = load()
GENRES = sorted({g for g, _ in SENTS})
GRADES = {"pooled": lambda k: True, "k even": lambda k: k % 2 == 0, "k odd": lambda k: k % 2 == 1,
          "k=0": lambda k: k == 0, "k=1": lambda k: k == 1, "k=2": lambda k: k == 2, "k>=3": lambda k: k >= 3}


INVERTIBLE = None


def invertible_tags(sents):
    """Tags whose round trip X -> B -> X on the POOLED data (all genres) returns to themselves:
    only for these is 'moved' meaningful (the layer map is many-to-one elsewhere, e.g. punctuation)."""
    X = sorted({t[0] for _, toks in sents for t in toks}); B = sorted({t[1] for _, toks in sents for t in toks})
    xi = {x: i for i, x in enumerate(X)}; bi = {b: i for i, b in enumerate(B)}
    N = np.zeros((len(X), len(B)))
    for _, toks in sents:
        for x, b, *_ in toks:
            N[xi[x], bi[b]] += 1
    T1 = N / N.sum(1, keepdims=True).clip(1); T2 = N.T / N.T.sum(1, keepdims=True).clip(1)
    L = T1 @ T2
    return {X[i] for i in range(len(X)) if L[i].argmax() == i and L[i, i] - np.partition(L[i], -2)[-2] > MARGIN}


def moved_count(sents, sel):
    """Sum over ordered genre pairs of the number of moved XPOS tags (invertible tags only)."""
    by = {g: [] for g in GENRES}
    for g, toks in sents:
        by[g].extend(t for t in toks if sel(t[2]))
    X = sorted({t[0] for _, toks in sents for t in toks}); B = sorted({t[1] for _, toks in sents for t in toks})
    xi = {x: i for i, x in enumerate(X)}; bi = {b: i for i, b in enumerate(B)}
    N = {}
    for g in GENRES:
        M = np.zeros((len(X), len(B)))
        for x, b, *_ in by[g]:
            M[xi[x], bi[b]] += 1
        N[g] = M
    total, detail = 0, []
    for g1 in GENRES:
        for g2 in GENRES:
            if g1 == g2:
                continue
            A = N[g1]; C = N[g2].T                     # C[b, x] counts in g2
            rows = A.sum(1)
            T1 = np.divide(A, rows[:, None], out=np.zeros_like(A), where=rows[:, None] > 0)
            crow = C.sum(1)
            T2 = np.divide(C, crow[:, None], out=np.zeros_like(C), where=crow[:, None] > 0)
            L = T1 @ T2
            for i in np.flatnonzero((rows >= MINC) & (N[g2].sum(1) >= MINC)):
                if X[i] not in INVERTIBLE:
                    continue
                j = int(L[i].argmax())
                if j != i and L[i, j] - L[i, i] > MARGIN:
                    total += 1
                    detail.append((g1, g2, X[i], X[j], round(float(L[i, j]), 3), round(float(L[i, i]), 3)))
    return total, detail


def shuffled(sents, rng):
    gs = [g for g, _ in sents]; rng.shuffle(gs)
    return [(g, toks) for g, (_, toks) in zip(gs, sents)]


def same_conventions(sents, rng):
    """Composition-preserving null: keep every word's B, form and arity (so each genre's composition,
    down to word forms, is unchanged), redraw its XPOS from the POOLED P(XPOS | B, form): identical
    conventions in all genres."""
    cond = collections.defaultdict(collections.Counter)
    for _, toks in sents:
        for x, b, _, w in toks:
            cond[(b, w)][x] += 1
    draw = {key: (list(c), np.array(list(c.values()), float) / sum(c.values())) for key, c in cond.items()}
    out = []
    for g, toks in sents:
        out.append((g, [(draw[(b, w)][0][rng.choice(len(draw[(b, w)][0]), p=draw[(b, w)][1])], b, k, w)
                        for _, b, k, w in toks]))
    return out


def planted(sents):
    out = []
    for g, toks in sents:
        if g == "email":
            toks = [("NNP" if (x == "NN" and k % 2 == 1) else x, b, k, w) for x, b, k, w in toks]
        out.append((g, toks))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    res = {}
    INVERTIBLE = invertible_tags(SENTS)
    print(f"invertible XPOS tags ({len(INVERTIBLE)}):", sorted(INVERTIBLE))
    print(f"EWT: {len(SENTS)} sentences, genres {GENRES}")
    print(f"{'grade':>7} | {'real':>5} | {'null mean':>9} {'null max':>8} | {'planted':>7}   (moved XPOS tags summed over 20 ordered genre pairs)")
    PL = planted(SENTS)
    for G, sel in GRADES.items():
        real, det = moved_count(SENTS, sel)
        nulls = [moved_count(shuffled(SENTS, rng), sel)[0] for _ in range(NNULL)]
        pl, pdet = moved_count(PL, sel)
        # composition-preserving null: which specific (g1, g2, tag -> tag) moves recur under it?
        comp = collections.Counter()
        for _ in range(NNULL):
            _, d = moved_count(same_conventions(SENTS, rng), sel)
            comp.update({m[:4] for m in d})
        unexplained = [m for m in det if comp[m[:4]] / NNULL < 0.05]
        pl_unexpl = [m for m in pdet if comp[m[:4]] / NNULL < 0.05]
        res[G] = dict(real=real, null_mean=float(np.mean(nulls)), null_max=int(max(nulls)), planted=pl,
                      real_unexplained=unexplained, planted_unexplained=pl_unexpl,
                      real_detail=det, planted_detail=pdet[:20])
        print(f"{G:>7} | {real:5d} | {np.mean(nulls):9.2f} {max(nulls):8d} | {pl:7d} | not explained by composition: "
              f"real {len(unexplained)}, planted {len(pl_unexpl)}")
    json.dump(res, open("examples/ewt_monodromy.json", "w"), indent=1)
    for G in GRADES:
        if res[G]["real_unexplained"]:
            print(f"\nreal moves NOT explained by composition, grade {G}:", res[G]["real_unexplained"][:12])
    print("\nplanted moves, grade 'k odd':", res["k odd"]["planted_detail"][:8])
