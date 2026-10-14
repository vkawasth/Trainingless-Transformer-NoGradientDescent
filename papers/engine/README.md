# Posterior Coherence Engine (`pce`)

Separates **uncertainty about the answer** from **failure of the evidence to admit a coherent global model**.
Report: `report/pce_report.pdf` (5 pp).

For a set of local reports the engine returns:

1. **Uncertainty**: the exact three-level split `H(rho) = between + within + noise` (`pce.split`).
2. **Coherence**: `coherent-unique`, `coherent-nonunique` (with fibre width), or `incoherent` (with consistency radius and a finite
   certificate found by Farkas duality) (`pce.coherence`).
3. **Pooling validity**: the Beck–Chevalley defect of a pooling rule and the KL-ball radius with its decision-error bound
   (`pce.defect`, `pce.kl_ball`).

`pce.decide` turns these into an action: answer, abstain, flag-incoherent, or flag-nonunique.

## Quick start

```bash
pip install numpy scipy scikit-learn matplotlib pandas
python check_pce.py                               # 6 checks, all pass
python -m pce.cli examples/three_token.json       # incoherent evidence, with its certificate
python demos/synthetic.py                         # synthetic demonstrations (debug and illustration only)
```

```python
from pce import Window, coherence, split, decide
import numpy as np
t = np.array([[0.125, 0.375], [0.375, 0.125]])
r = coherence({"X0": 2, "X1": 2, "X2": 2},
              [Window(p, t) for p in [("X0", "X1"), ("X1", "X2"), ("X0", "X2")]])
print(r.status, r.radius, r.violation)            # incoherent 0.083 0.375
```

## Contents

| Path | What it is |
|---|---|
| `pce/` | the package: `coherence.py`, `uncertainty.py`, `pooling.py`, `decide.py`, `votes.py`, `selective.py`, `cli.py` |
| `check_pce.py` | exact checks against the paper (three-token model, certificates, split, defect, KL ball, decisions) |
| `demos/synthetic.py` | D1 averaging hides conflicts; D2 non-unique reconstruction; D3 where disagreement sits (with a null world) |
| `study4/` | pre-registered real-data test: `PREREG_engine.md` (frozen, SHA-256 in `FROZEN_PREREG_engine.txt`), `study4_analysis.py`, results |
| `report/` | `pce_report.tex/.pdf` and `fig_engine.py` |

## Study 4 (pre-registered)

LLM-drift answers of GPT-4 and GPT-3.5 (March and June) on five tasks, 2358 items. Adding where disagreement sits (between versus
within model families) to the averaging baseline improves the prediction of whether the pooled answer is correct by
**0.0175 bits per item, 95% interval [0.0107, 0.0241]**: the pre-registered hypothesis is supported. The gain is concentrated in primes
and happy numbers, where model versions drifted.

```bash
cd study4 && LLMDRIFT_DIR=/path/to/llmdrift python study4_analysis.py
```

The incoherence detector is not tested on real data yet: each item's answers are observed jointly, and jointly observed tables always
admit a global law. Pairwise elicitation (`../halluc_study/kit_pairwise_cycles.py`) collects the data that would test it.
