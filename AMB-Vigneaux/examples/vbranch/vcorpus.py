"""Build Corpus V (variable right-branching, fixed depth) and its exact diagnostics."""
import json, numpy as np, collections
from vtree import *

g = random_grammar(L=3, V=8, A=12, m=3, kmax=5, seed=0)
rng = np.random.default_rng(1)
tr, trees = sample(g, 20000, rng, return_trees=True)
va = sample(g, 2000, rng)
json.dump(dict(grammar=g.to_json(), train=tr, val=va), open("cV/corpus.json", "w"))

def n_trees(s):
    """number of derivations of s (inside with all probabilities 1, prior 1)."""
    g1 = VGrammar(g.L, g.V, g.A, {l: [(b, c, 1.0) for b, c, _ in rs] for l, rs in g.rules.items()}, np.ones(g.V))
    return np.exp(chart_io(g1, s)[0])

ll = np.array([chart_io(g, s)[0] for s in va]); lens = np.array([len(s) for s in va])
amb = np.array([n_trees(s) for s in va[:500]])
arity = collections.Counter(len(c) for l, rs in g.rules.items() for _, c, _ in rs)
print(f"productions by number of children (1 + k): {dict(sorted(arity.items()))}")
print(f"sentence length: mean {lens.mean():.1f}, median {np.median(lens):.0f}, min {lens.min()}, max {lens.max()}")
print(f"exact held-out log p per sentence {ll.mean():.3f}, per token {ll.sum()/lens.sum():.4f} nats")
print(f"derivations per sentence (500 val): 1 -> {np.mean(amb==1):.3f}, max {amb.max():.0f}")
json.dump(dict(ll_sent=float(ll.mean()), ll_tok=float(ll.sum()/lens.sum()), len_mean=float(lens.mean()),
               frac_unambiguous=float(np.mean(amb == 1)), max_derivations=float(amb.max()),
               arity=dict(sorted(arity.items()))), open("cV/diagnostics.json", "w"), indent=1)
