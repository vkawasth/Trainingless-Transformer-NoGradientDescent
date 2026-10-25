"""The decision layer: answer, abstain, or flag.

The engine keeps two questions apart:
  * how uncertain is the answer?        -> the three-level split (uncertainty.split)
  * does the evidence admit one model?  -> coherence (coherence.coherence)

decide() turns a report into an action. Incoherent evidence is flagged (with its certificate) rather than averaged; non-unique
reconstructions are reported with their width; otherwise the pooled answer is given, or withheld when a confidence score falls below
a threshold.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .uncertainty import Split
from .coherence import CoherenceReport


@dataclass
class Decision:
    action: str            # "answer", "abstain", "flag-incoherent", "flag-nonunique"
    answer: int | None
    reason: str


def decide(split: Split | None = None, coh: CoherenceReport | None = None, score: float | None = None,
           threshold: float = 0.5, radius_tol: float = 0.0, width_tol: float = np.inf) -> Decision:
    if coh is not None and coh.status == "incoherent" and coh.radius > radius_tol:
        return Decision("flag-incoherent", None, f"no global law; certificate violated by {coh.violation:.3f}")
    if coh is not None and coh.status == "coherent-nonunique" and coh.width > width_tol:
        return Decision("flag-nonunique", None, f"global law not determined; width {coh.width:.3f}")
    ans = int(np.argmax(split.rho)) if split is not None else None
    if score is not None and score < threshold:
        return Decision("abstain", ans, f"confidence {score:.3f} below {threshold}")
    return Decision("answer", ans, "coherent evidence, confident")
