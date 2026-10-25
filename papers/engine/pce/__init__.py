"""Posterior Coherence Engine (pce).

Separates uncertainty about the answer from failure of the evidence to admit a coherent global model.
Companion code to "Shadows, Fibres and Secants of the Posterior Tower" and "Pooling Informed Agents".
"""
from .coherence import Window, CoherenceReport, coherence, pair_window
from .uncertainty import split, Split, H
from .pooling import defect, markov_glue, kl_ball
from .decide import decide, Decision
from .integrate import (structure, gluable_fraction, maxent_merge, irreconcilable, min_splits, all_min_splits, integrate,
                        Integration, Gluable)

__all__ = ["Window", "CoherenceReport", "coherence", "pair_window", "split", "Split", "H",
           "defect", "markov_glue", "kl_ball", "decide", "Decision",
           "structure", "gluable_fraction", "maxent_merge", "irreconcilable", "min_splits", "all_min_splits", "integrate",
           "Integration", "Gluable"]
__version__ = "0.1.0"
