"""Command line: python -m pce.cli evidence.json

evidence.json:
  {"domains": {"X0": 2, "X1": 2, "X2": 2},
   "windows": [{"vars": ["X0", "X1"], "table": [[0.1, 0.4], [0.4, 0.1]]}, ...],
   "groups": [[[1, [0.9, 0.1]], [1, [0.6, 0.4]]], [[1, [0.2, 0.8]]]]}      (optional: hierarchical posteriors for the split)
Prints the coherence report, the three-level split and the decision.
"""
import json, sys
import numpy as np
from . import Window, coherence, split, decide


def main(path):
    ev = json.load(open(path))
    coh = None
    if "windows" in ev:
        wins = [Window(tuple(w["vars"]), np.array(w["table"], float)) for w in ev["windows"]]
        coh = coherence(ev["domains"], wins)
        print(f"coherence: {coh.status}; radius {coh.radius:.4f}; width {coh.width:.4f}")
        if coh.certificate is not None:
            print(f"certificate violated by {coh.violation:.4f} (every global law gives >= 0):")
            for i, arr in coh.certificate.items():
                print(f"  window {wins[i].vars}: {np.round(arr, 4).tolist()}")
    sp = None
    if "groups" in ev:
        sp = split(ev["groups"])
        print(f"split (bits): total {sp.total:.4f} = between {sp.between:.4f} + within {sp.within:.4f} + noise {sp.noise:.4f}")
    d = decide(sp, coh, score=None if sp is None else 1 - sp.total / max(np.log2(len(sp.rho)), 1e-12))
    print(f"decision: {d.action} ({d.reason})")


if __name__ == "__main__":
    main(sys.argv[1])
