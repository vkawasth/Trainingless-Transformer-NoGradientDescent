"""BASIL: relational claims ("A said X", "C caused D") per outlet, conflicts across outlets per event.
usage: python examples/basil_relational.py /path/to/BASIL"""
import sys, json, glob, collections
from amb_vigneaux.relclaims import extract, detect, overlap_stats

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/BASIL"
events = collections.defaultdict(dict)
for f in glob.glob(f"{root}/articles/*/*.json"):
    d = json.load(open(f))
    text = " ".join(s for para in d["body-paragraphs"] for s in para)
    events[d["triplet-uuid"]][d["source"].lower()] = text
events = {t: v for t, v in events.items() if len(v) == 3}
stats = collections.Counter(); per_kind = collections.Counter(); flagged = []; ov = collections.Counter()
for t, arts in sorted(events.items()):
    cl = [c for src, txt in arts.items() for c in extract(src, txt)]
    for c in cl:
        stats[(c.source, c.kind)] += 1
    for k, v in overlap_stats(cl).items():
        ov[k] += v
    for x in detect(cl):
        per_kind[x.kind] += 1
        flagged.append((t, x))
print(f"events: {len(events)}")
print("claims extracted (outlet, kind):", dict(sorted(stats.items())))
print("shared claims (made by >= 2 outlets):", dict(ov))
print("conflicts by kind:", dict(per_kind))
out = []
for t, x in flagged:
    item = dict(event=t, kind=x.kind, sources=list(x.sources),
                claims=[dict(source=c.source, kind=c.kind, a=c.a, b=c.b if c.kind == "cause" else sorted(c.content_words),
                             polarity=c.polarity, sentence=c.sentence[:220]) for c in x.claims])
    out.append(item)
json.dump(dict(n_events=len(events), claims={f"{k[0]}:{k[1]}": v for k, v in stats.items()},
               conflicts=dict(per_kind), overlap=dict(ov), flagged=out), open("examples/basil_relational.json", "w"), indent=1)
