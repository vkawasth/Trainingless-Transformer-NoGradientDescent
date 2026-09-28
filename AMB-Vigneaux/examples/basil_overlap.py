"""BASIL (Fan et al., EMNLP 2019), overlap channel only, gold annotations only.

Sources = outlets (fox, nyt, hpo); 100 events, each covered by all three.
Propositions = "the article is negative toward target T in event t" for T = main event or
an annotated entity.  Two readings:
  A  article-level author feeling (neg -> 1, pos -> 0, neu -> no claim): deterministic claims
  B  phrase-level biased spans toward T (neg span -> 1, pos span -> 0): sampled claims, pooled noise
Literals are encoded as parity edges to an anchor proposition 0 ("x_T XOR TRUE = value"),
so an overlap disagreement is two outlets asserting different parities on the same edge.
Cycles cannot arise from literals alone; this run exercises the overlap channel only.
Null: permute outlet labels within each event (1000 permutations).

usage: python examples/basil_overlap.py /path/to/BASIL
"""
import sys, json, glob, itertools, collections, random
from amb_vigneaux.meter import run_meter

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/BASIL"
arts = {}
for f in glob.glob(f"{root}/articles/*/*.json"):
    d = json.load(open(f)); arts[d["uuid"]] = d
events = collections.defaultdict(dict)
for f in glob.glob(f"{root}/annotations/*/*.json"):
    d = json.load(open(f))
    a = arts.get(d["uuid"])
    if a is None:
        continue
    events[a["triplet-uuid"]][a["source"].lower()] = d
events = {t: v for t, v in events.items() if set(v) == {"fox", "nyt", "hpo"}}
OUT = ("fox", "nyt", "hpo")
PAIRS = list(itertools.combinations(OUT, 2))
STANCE = {"left": "L", "liberal": "L", "center": "C", "right": "R", "conservative": "R"}


def claims_article(ev, labels):
    """Reading A. labels maps outlet -> annotation dict (possibly permuted)."""
    props, cl = {}, []
    for t, byo in ev.items():
        for o in OUT:
            for k, v in labels[t][o]["article-level-annotations"].items():
                if not k.startswith("author_feeling_") or v not in ("neg", "pos"):
                    continue
                pid = props.setdefault((t, k[len("author_feeling_"):]), len(props) + 1)
                cl.append((o, 0, pid, 1 if v == "neg" else 0))
    return cl, props


def claims_phrase(ev, labels):
    """Reading B: one sampled claim per biased span with a target."""
    props, cl = {}, []
    for t in ev:
        for o in OUT:
            for p in labels[t][o]["phrase-level-annotations"]:
                if p["polarity"] not in ("neg", "pos") or not p["target"]:
                    continue
                pid = props.setdefault((t, p["target"]), len(props) + 1)
                cl.append((o, 0, pid, 1 if p["polarity"] == "neg" else 0))
    return cl, props


def pair_stats(claims, deterministic):
    rep = run_meter(claims, deterministic=deterministic, noise="pooled")
    # shared propositions per pair (both outlets made at least one claim)
    have = collections.defaultdict(set)
    for o, _, pid, _ in claims:
        have[o].add(pid)
    shared = {p: len(have[p[0]] & have[p[1]]) for p in PAIRS}
    dis = collections.Counter(tuple(sorted(d["units"], key=OUT.index)) for d in rep.overlap_disagreements)
    rate = {p: dis[p] / shared[p] if shared[p] else float("nan") for p in PAIRS}
    return rate, dis, shared, rep


def separation(rate):
    """fox-vs-others minus nyt-vs-hpo."""
    return (rate[("fox", "nyt")] + rate[("fox", "hpo")]) / 2 - rate[("nyt", "hpo")]


rng = random.Random(0)
results = {}
for name, builder, det in (("A: article-level author feeling", claims_article, True),
                           ("B: phrase-level biased spans", claims_phrase, False)):
    labels = {t: dict(v) for t, v in events.items()}
    cl, props = builder(events, labels)
    rate, dis, shared, rep = pair_stats(cl, det)
    obs = separation(rate)
    null = []
    for _ in range(1000):
        perm = {}
        for t, v in events.items():
            docs = [v[o] for o in OUT]; rng.shuffle(docs)
            perm[t] = dict(zip(OUT, docs))
        c2, _ = builder(events, perm)
        null.append(separation(pair_stats(c2, det)[0]))
    p = (1 + sum(x >= obs for x in null)) / (1 + len(null))
    # stance check: do disagreeing article pairs differ in annotated relative stance more often?
    diff_st = collections.Counter()
    for d in rep.overlap_disagreements:
        (t, _), = [k for k, v in props.items() if v == d["edge"][1]]
        a, b = d["units"]
        sa = STANCE.get(events[t][a]["article-level-annotations"].get("relative_stance"), "?")
        sb = STANCE.get(events[t][b]["article-level-annotations"].get("relative_stance"), "?")
        diff_st["different stance" if sa != sb else "same stance"] += 1
    results[name] = dict(n_props=len(props), n_claims=len(cl), eps_hat=rep.eps_hat,
                         shared={"-".join(k): v for k, v in shared.items()},
                         disagreements={"-".join(k): dis[k] for k in PAIRS},
                         rate={"-".join(k): round(v, 3) for k, v in rate.items()},
                         separation=round(obs, 3), perm_p=round(p, 4), stance_of_disagreeing_pairs=dict(diff_st))
    print(f"\n{name}\n  propositions {len(props)}, claims {len(cl)}"
          + (f", pooled flip rate {rep.eps_hat:.3f}" if rep.eps_hat is not None else ""))
    for k in PAIRS:
        print(f"  {k[0]}-{k[1]}: shared {shared[k]:>4}, disagreements {dis[k]:>3}, rate {rate[k]:.3f}")
    print(f"  separation (fox-vs-others minus nyt-hpo) = {obs:.3f}, permutation p = {p:.4f}")
    print(f"  disagreeing pairs by annotated relative stance: {dict(diff_st)}")

# baseline for the stance check: all article pairs within events
base = collections.Counter()
for t, v in events.items():
    for a, b in PAIRS:
        sa = STANCE.get(v[a]["article-level-annotations"].get("relative_stance"), "?")
        sb = STANCE.get(v[b]["article-level-annotations"].get("relative_stance"), "?")
        base["different stance" if sa != sb else "same stance"] += 1
print(f"\nbaseline, all within-event article pairs: {dict(base)}   (events used: {len(events)})")
results["baseline_pairs"] = dict(base); results["n_events"] = len(events)
json.dump(results, open("examples/basil_overlap.json", "w"), indent=1)
