"""Study 82 step 1b EXPLORATORY (no pass/fail): continuity of generators with the AMEND1 core placement, ladder 0.01 ... 0.02."""
import os, sys, pickle, itertools, numpy as np
E = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, E)
from check_wall81 import wrapdist
OUT = open(os.path.join(E, "study82_path", "dump82b.txt"), "a")
def say(s): print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
TS = ["0.01", "0.015", "0.01525", "0.0155", "0.01575", "0.016", "0.01625", "0.0165", "0.01675", "0.017", "0.01725", "0.0175", "0.02"]
P = {t: pickle.load(open(os.path.join(E, "study82_path", f"t{t}", "b82.pkl"), "rb")) for t in TS}
for t in TS: say(f"t = {t}: dims {[P[t]['dims'][k] for k in itertools.combinations(range(6), 2)]}; mu1 {P[t]['mu1']}; words unchanged {P[t]['words_ok']}")
for t1, t2 in zip(TS[:-1], TS[1:]):
    worst = []
    for pq in itertools.combinations(range(6), 2):
        z1, z2 = P[t1]["pts"][pq], P[t2]["pts"][pq]
        if not z1 or len(z1) != len(z2): continue
        Dm = np.array([[wrapdist(a, b, P[t2]["wa"], P[t2]["wb"]) for b in z2] for a in z1]); m = Dm.argmin(1)
        sep = min([wrapdist(a, b, P[t1]["wa"], P[t1]["wb"]) for a, b in itertools.combinations(z1, 2)] + [wrapdist(a, b, P[t2]["wa"], P[t2]["wb"]) for a, b in itertools.combinations(z2, 2)] + [np.inf])
        worst.append((round(float(Dm[np.arange(len(m)), m].max() / (0.5 * sep)), 4), [q + 1 for q in pq], len(set(m.tolist())) == len(m)))
    worst.sort(reverse=True); say(f"{t1}->{t2}: worst displacement / half-separation {worst[:2]}")
