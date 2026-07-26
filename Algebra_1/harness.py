"""
Shared harness for the intrinsic-dimension probe.

Loads the compiler's model/get_batch definitions against a chosen corpus by
writing the /tmp corpus files the prefix expects, then exec'ing the prefix up to
the SPECTRAL E0 block (not up to PHASE 1).

Why the shorter cut: everything between "# CORPUS + SPECTRAL E0" and "# PHASE 1"
is (a) a 257-eigenvector eigsh on the VOCABxVOCAB bigram Laplacian and (b) a
200-step floor-gradient training run. (a) is impossible at VOCAB<=256 (eigsh
needs k<N) and, more importantly, it makes the *initialisation* a function of
the corpus bigram graph -- which would leak corpus structure into theta_0 and
confound a cross-corpus probe. (b) is unused here. Cut at line 147 gives the
class definitions, get_batch, eval_val and the constants, with no corpus
leakage and an identical seeded init across all corpora.
"""
import json, os, torch, numpy as np

SRC = open("/home/claude/work/compiler_geometri_patched_86.py").read()
CUT = SRC.find("# \u2500\u2500 CORPUS + SPECTRAL E\u2080")
assert CUT > 0, "cut marker not found"

CORPORA = "/home/claude/work/bundle/corpora"
VOCAB_FIXED = 256          # shared across all corpora so the model is identical
SEED_INIT = 99             # the compiler's model seed


def load_corpus(name, vocab=VOCAB_FIXED, val_frac=0.1):
    ids = json.load(open(f"{CORPORA}/{name}.json"))
    ids = [int(t) % vocab for t in ids]
    n_val = int(len(ids) * val_frac)
    train, val = ids[:-n_val], ids[-n_val:]
    json.dump(train, open("/tmp/train_ids.json", "w"))
    json.dump(val, open("/tmp/val_ids.json", "w"))
    json.dump(list(range(vocab)), open("/tmp/vocab.json", "w"))
    return len(ids)


def build(name, seed_init=SEED_INIT):
    """Returns (globals_dict, model, get_batch). Model init is corpus-independent."""
    load_corpus(name)
    g = {}
    exec(SRC[:CUT], g)
    torch.manual_seed(seed_init)
    model = g["LM"]()          # standard normal init; NO spectral E_init
    return g, model, g["get_batch"]


def param_groups(model):
    """Index ranges into the flat parameter vector, by role."""
    named = list(model.named_parameters())
    idx = {}
    i = 0
    for nm, p in named:
        k = p.numel()
        idx[nm] = (i, i + k)
        i += k
    P = i
    block = np.zeros(P, dtype=bool)
    emb = np.zeros(P, dtype=bool)
    for nm, (a, b) in idx.items():
        if nm.startswith("blocks."):
            block[a:b] = True
        elif nm == "te.weight":
            emb[a:b] = True
    return P, block, emb, named
