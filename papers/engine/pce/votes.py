"""From grouped categorical votes to the engine's features.

Each item has answers from several runs, grouped into families (for example model families). Runs are treated as one-hot posteriors,
so the noise term of the split is zero and total = between + within.

  pooled_answer(answers, tiebreak): plurality answer; ties broken by the run listed in `tiebreak` when it is among the tied answers,
      otherwise by the smallest answer in sorted string order.
  features(answers, families, tiebreak): dict with the pooled answer, its vote share v, total entropy H(rho) (bits), the between-family
      term I(answer; family) and the within-family term.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from .uncertainty import split


def pooled_answer(answers: dict, tiebreak: str | None = None):
    counts = Counter(answers.values()); top = max(counts.values())
    tied = sorted((a for a, c in counts.items() if c == top), key=str)
    if tiebreak is not None and answers.get(tiebreak) in tied: return answers[tiebreak]
    return tied[0]


def features(answers: dict, families: dict, tiebreak: str | None = None):
    labels = sorted(set(answers.values()), key=str); idx = {a: i for i, a in enumerate(labels)}
    groups = []
    for fam, runs in families.items():
        members = [(1.0, np.eye(len(labels))[idx[answers[r]]]) for r in runs if r in answers]
        if members: groups.append(members)
    sizes = np.array([len(g) for g in groups], float)
    s = split(groups, sizes / sizes.sum())
    ans = pooled_answer(answers, tiebreak)
    v = sum(1 for a in answers.values() if a == ans) / len(answers)
    return {"answer": ans, "v": v, "total": s.total, "between": s.between, "within": s.within}
