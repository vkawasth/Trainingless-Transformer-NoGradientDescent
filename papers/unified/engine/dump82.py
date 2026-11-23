"""Study 82 step 1 EXPLORATORY (no pass/fail): generator displacements, class-word lengths and taut pegs on the dense ladder [0.015, 0.0175]."""
import os, sys, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from check_wall81 import wrapdist
E = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = open(os.path.join(E, "study82_path", "dump82.txt"), "a")
def say(s): print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
TS = ["0.015", "0.01525", "0.0155", "0.01575", "0.016", "0.01625", "0.0165", "0.01675", "0.017", "0.01725", "0.0175"]
P = {t: pickle.load(open(os.path.join(E, "study81_wall", f"t{t}", "points81.pkl"), "rb")) for t in TS}
B = {t: pickle.load(open(os.path.join(E, "study81_wall", f"t{t}", "bside81.pkl"), "rb")) for t in TS}
for t in TS: say(f"t = {t}: word lengths {[len(w) for w in B[t]['words']]}; pegs {[sum(1 for q in c['pegs'] if q is not None) for c in B[t]['taut']]}; dims {[B[t]['dims'][k] for k in itertools.combinations(range(6), 2)]}")
for t1, t2 in zip(TS[:-1], TS[1:]):
    worst = []
    for pq in itertools.combinations(range(6), 2):
        z1, z2 = P[t1]["pts"][pq], P[t2]["pts"][pq]
        if not z1 or len(z1) != len(z2): continue
        Dm = np.array([[wrapdist(a, b, P[t2]["wa"], P[t2]["wb"]) for b in z2] for a in z1]); m = Dm.argmin(1)
        sep = min([wrapdist(a, b, P[t1]["wa"], P[t1]["wb"]) for a, b in itertools.combinations(z1, 2)] + [np.inf])
        worst.append((Dm[np.arange(len(m)), m].max() / (0.5 * sep), [q + 1 for q in pq], round(float(Dm[np.arange(len(m)), m].max()), 5), len(set(m.tolist())) == len(m)))
    worst.sort(reverse=True); say(f"{t1}->{t2}: worst displacement / half-separation {[w for w in worst[:3]]}")
