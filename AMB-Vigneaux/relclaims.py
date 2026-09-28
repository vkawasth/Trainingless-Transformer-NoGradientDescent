"""Relational claims: "A said X" and "C caused D", extracted per source, checked across sources.

Extraction (spaCy dependency parse, transparent rules):
  attribution   verb in SAY (say, tell, claim, deny, ...) with a nominal subject = speaker and a clausal
                complement or quotation = content;  "according to A, X";  deny/reject flip the polarity.
  causation     verb in CAUSE (cause, lead to, result in, trigger, spark, prompt, fuel, force, ...) with
                subject = cause, object / "to"/"in"-object = effect; passive "Y was caused by X";
                "Y because of / due to X".  A negated verb flips the polarity.
Normalisation: an argument is the lemma of its head noun with compound modifiers ("tax cut"); a speaker
is the proper-noun chain of the subject; contents are compared by content-word overlap (Jaccard).

Propositions and axioms (stated, not hidden):
  Said(A, c) with polarity;   Cause(C, D) with polarity.
  A1  a statement has one speaker:            not (Said(A, c) and Said(B, c)) for A != B
  A2  causation is not mutual:                not (Cause(C, D) and Cause(D, C))
  A3  no causal 3-cycles:                     not (Cause(C, D) and Cause(D, E) and Cause(E, C))
Detected conflicts, each with its sources and the sentences involved:
  polarity   the same proposition asserted and denied (by two sources = overlap; by one = self)
  speaker    one content attributed to different speakers (A1)
  reversal   Cause(C, D) against Cause(D, C) (A2)
  cycle      a causal 3-cycle assembled from two or more sources (A3; the cyclic type)
These are FLAGS: a repeated quote, a paraphrase or a genuine feedback loop can trigger A1-A3.
"""
from __future__ import annotations

import collections, itertools, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

SAY = {"say", "tell", "claim", "argue", "state", "insist", "add", "note", "write", "announce", "allege",
       "warn", "acknowledge", "suggest", "contend", "assert", "declare", "deny", "reject", "dispute"}
NEG_SAY = {"deny", "reject", "dispute"}
CAUSE = {"cause", "trigger", "spark", "prompt", "fuel", "force", "drive", "produce", "provoke", "create"}
CAUSE_PREP = {"lead": "to", "result": "in", "contribute": "to", "give": "rise"}
STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "that", "this", "is", "was", "be", "it", "he", "she",
        "they", "we", "for", "on", "with", "as", "at", "by", "his", "her", "their", "its", "has", "have", "had",
        "will", "would", "not", "no", "but", "from", "are", "were", "been", "who", "which", "what", "said"}
PRON = {"he", "she", "they", "it", "we", "i", "you", "who", "this", "that"}
# speakers that are roles, documents or channels: unresolved without coreference, never used for A1
ROLE = {"president", "attorney", "lawyer", "spokesman", "spokeswoman", "spokesperson", "official", "officials",
        "aide", "aides", "leader", "source", "sources", "senator", "congressman", "ex", "governor", "secretary",
        "chairman", "director", "hopeful", "candidate", "statement", "letter", "report", "memo", "excerpts",
        "times", "post", "journal", "news", "network", "campaign", "committee", "office", "department",
        "white house", "administration", "a.g.", "ag", "group", "people", "critic", "critics", "supporter",
        "supporters", "democrat", "democrats", "republican", "republicans", "analyst", "analysts", "expert", "experts"}

_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class RelClaim:
    source: str
    kind: str                   # "said" | "cause"
    a: str                      # speaker, or cause
    b: str                      # content key, or effect
    polarity: bool
    sentence: str
    content_words: frozenset = field(default_factory=frozenset)


def _phrase(tok) -> str:
    """Head lemma with compound / proper-noun modifiers, lower-cased."""
    parts = [c for c in tok.children if c.dep_ in ("compound", "amod") and c.i < tok.i]
    words = [c.lemma_.lower() for c in sorted(parts, key=lambda t: t.i)] + [tok.lemma_.lower()]
    return " ".join(w for w in words if w.isalpha())


def _in_quote(tok) -> bool:
    """True if the token lies between an odd number of quotation marks earlier in its sentence."""
    q = sum(1 for t in tok.sent if t.i < tok.i and t.text in ('"', "\u201c", "\u201d", "''", "``"))
    return q % 2 == 1


def _speaker(tok) -> Optional[str]:
    if tok.lower_ in PRON or tok.pos_ == "PRON" or _in_quote(tok):
        return None
    parts = [c for c in tok.children if c.dep_ in ("compound", "flat") and c.i < tok.i] + [tok]
    name = " ".join(t.text for t in sorted(parts, key=lambda t: t.i))
    return name.lower() if any(t.pos_ in ("PROPN", "NOUN") for t in parts) else None


def _content_words(span) -> frozenset:
    return frozenset(t.lemma_.lower() for t in span if t.is_alpha and t.lemma_.lower() not in STOP and not t.is_stop)


def _negated(tok) -> bool:
    return any(c.dep_ == "neg" for c in tok.children)


def extract(source: str, text: str) -> List[RelClaim]:
    out = []
    doc = nlp()(text)
    for sent in doc.sents:
        s = sent.text.strip()
        for tok in sent:
            lem = tok.lemma_.lower()
            # ---- attribution
            nested = any(a.lemma_.lower() in SAY and a.pos_ == "VERB" for a in tok.ancestors)
            if tok.pos_ == "VERB" and lem in SAY and not nested:
                subj = next((c for c in tok.children if c.dep_ in ("nsubj",)), None)
                comp = next((c for c in tok.children if c.dep_ in ("ccomp", "xcomp", "parataxis")), None)
                if comp is None and tok.head is not tok and tok.dep_ in ("parataxis",):
                    comp = tok.head
                if subj is not None and comp is not None:
                    spk = _speaker(subj)
                    cw = _content_words(comp.subtree)
                    if spk and len(cw) >= 3:
                        pol = (lem not in NEG_SAY) != _negated(tok)
                        out.append(RelClaim(source, "said", spk, "", pol, s, cw))
            if tok.lower_ == "according" and tok.dep_ == "prep":
                to = next((c for c in tok.children if c.lower_ == "to"), None)
                obj = next((c for c in to.children if c.dep_ == "pobj"), None) if to is not None else None
                if obj is not None:
                    spk = _speaker(obj)
                    cw = _content_words([t for t in sent if t not in set(tok.subtree)])
                    if spk and len(cw) >= 3:
                        out.append(RelClaim(source, "said", spk, "", True, s, cw))
            # ---- causation
            if tok.pos_ == "VERB" and (lem in CAUSE or lem in CAUSE_PREP):
                subj = next((c for c in tok.children if c.dep_ == "nsubj"), None)
                agent = next((c for c in tok.children if c.dep_ == "agent"), None)
                subjpass = next((c for c in tok.children if c.dep_ == "nsubjpass"), None)
                obj = next((c for c in tok.children if c.dep_ in ("dobj", "attr")), None)
                if lem in CAUSE_PREP:
                    prep = next((c for c in tok.children if c.dep_ == "prep" and c.lower_ == CAUSE_PREP[lem]), None)
                    obj = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep is not None else obj
                cause = effect = None
                if subj is not None and obj is not None:
                    cause, effect = subj, obj
                elif agent is not None and subjpass is not None:
                    by = next((c for c in agent.children if c.dep_ == "pobj"), None)
                    cause, effect = by, subjpass
                if cause is not None and effect is not None and cause.pos_ != "PRON" and effect.pos_ != "PRON":
                    c, e = _phrase(cause), _phrase(effect)
                    if c and e and c != e:
                        out.append(RelClaim(source, "cause", c, e, not _negated(tok), s))
            if tok.lower_ in ("because", "due") and tok.dep_ in ("prep", "mark", "amod", "advmod"):
                pobj = next((c for c in tok.subtree if c.dep_ == "pobj"), None)
                root = sent.root
                eff = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                if pobj is not None and eff is not None and pobj.pos_ != "PRON" and eff.pos_ != "PRON":
                    c, e = _phrase(pobj), _phrase(eff)
                    if c and e and c != e:
                        out.append(RelClaim(source, "cause", c, e, not _negated(root), s))
    return out


def _jacc(a: frozenset, b: frozenset) -> float:
    return len(a & b) / max(1, len(a | b))


def cluster_contents(claims: Sequence[RelClaim], thr: float = 0.5) -> None:
    """Assign content keys to attribution claims by single-link clustering on word overlap."""
    said = [c for c in claims if c.kind == "said"]
    key = list(range(len(said)))

    def find(i):
        while key[i] != i:
            key[i] = key[key[i]]; i = key[i]
        return i
    for i, j in itertools.combinations(range(len(said)), 2):
        if _jacc(said[i].content_words, said[j].content_words) >= thr:
            key[find(i)] = find(j)
    for i, c in enumerate(said):
        c.b = f"content#{find(i)}"


@dataclass
class Conflict:
    kind: str
    sources: Tuple[str, ...]
    claims: List[RelClaim]


def overlap_stats(claims: Sequence[RelClaim]):
    """How often two sources make the SAME claim (the opportunity for a contradiction)."""
    cluster_contents(claims)
    said = collections.defaultdict(set); cause = collections.defaultdict(set)
    for c in claims:
        if c.kind == "said":
            said[c.b].add(c.source)
        else:
            cause[frozenset((c.a, c.b))].add(c.source)
    return dict(said_shared=sum(len(v) >= 2 for v in said.values()), said_total=len(said),
                cause_shared=sum(len(v) >= 2 for v in cause.values()), cause_total=len(cause))


def detect(claims: Sequence[RelClaim]) -> List[Conflict]:
    cluster_contents(claims)
    out = []
    by_prop: Dict[Tuple, List[RelClaim]] = {}
    for c in claims:
        by_prop.setdefault((c.kind, c.a, c.b), []).append(c)
    # polarity
    for (kind, a, b), cs in by_prop.items():
        pos = [c for c in cs if c.polarity]; neg = [c for c in cs if not c.polarity]
        for p, n in itertools.product(pos, neg):
            out.append(Conflict("polarity" + ("/self" if p.source == n.source else ""), (p.source, n.source), [p, n]))
    # speaker (A1): one content, different speakers, both asserted
    by_content: Dict[str, List[RelClaim]] = {}
    for c in claims:
        if c.kind == "said" and c.polarity:
            by_content.setdefault(c.b, []).append(c)
    def resolved(sp):
        w = [t for t in re.split(r"[\s\-]+", sp) if t]
        TITLES = {"mr.", "mrs.", "ms.", "dr.", "rep.", "sen.", "gov.", "the"}
        return sp not in ROLE and not all(t in ROLE or t in TITLES for t in w)

    for key, cs in by_content.items():
        for x, y in itertools.combinations(cs, 2):
            if (x.a != y.a and not (set(x.a.split()) & set(y.a.split())) and x.source != y.source
                    and resolved(x.a) and resolved(y.a)):
                out.append(Conflict("speaker", (x.source, y.source), [x, y]))
    # reversal (A2) and 3-cycles (A3) over asserted causal claims
    cause = [c for c in claims if c.kind == "cause" and c.polarity]
    edges: Dict[Tuple[str, str], List[RelClaim]] = {}
    for c in cause:
        edges.setdefault((c.a, c.b), []).append(c)
    for (x, y), cs in edges.items():
        if (y, x) in edges and x < y:
            for p, q in itertools.product(cs, edges[(y, x)]):
                out.append(Conflict("reversal" + ("/self" if p.source == q.source else ""), (p.source, q.source), [p, q]))
    nodes = sorted({n for e in edges for n in e})
    for a, b, c in itertools.permutations(nodes, 3):
        if a < b and a < c and (a, b) in edges and (b, c) in edges and (c, a) in edges:
            trio = [edges[(a, b)][0], edges[(b, c)][0], edges[(c, a)][0]]
            srcs = tuple(sorted({t.source for t in trio}))
            if len(srcs) >= 2:
                out.append(Conflict("cycle", srcs, trio))
    return out
