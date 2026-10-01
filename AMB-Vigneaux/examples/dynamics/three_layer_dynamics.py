"""Three-layer dynamics: a synthetic generator with planted phase changes, and which invariant sees which.

Generative model (two periods: corpus 1 = before the event t*, corpus 2 = after).
  Layer 1, outcomes: each event e has a topic t_e in {0,1,2} and a latent outcome o_e in {0,1,2}
      with P(o) = (0.6, 0.25, 0.15). Each event has 4 ITEMS (claims); item i has a structural ARITY
      k_i in {2,3,4,5} (clauses / entities) and a true label y_i = o_e w.p. 0.8, else uniform.
  Layer 2, sources A, B, C: each covers each event w.p. 0.8 (coverage creates arity GRADES of units).
      A source reports item labels through its own fixed relabelling pi_s (noise 0.1), and a stance score
          b(s,e) = u_s + v_{t_e} + h(s, t_e) + N(0, 1),     u = (0, 0.5, -0.5), v = (0, 0.3, -0.3), h = 0.
  Layer 3: relations among biases (loops) and among label maps (monodromy).
PLANTED PHASE CHANGES at t* (one at a time; 'none' is the null):
  A  outcome inversion         after t*: P(o) = (0.25, 0.6, 0.15)          (layer 1)
  B  natural lean shift         after t*: u_B += 1                          (uniform: a natural transformation)
  C  non-natural shift          after t*: h(B, topic 0) += 1                (topic-selective: naturality fails)
  D  monodromy twist            after t*: source C cyclically relabels ODD-arity items
  E  coverage-graded shift      after t*: b(C, e) += 1 only on events where A is missing
DETECTORS (each run on corpus 1 vs corpus 2):
  d1 outcome distribution       chi-square of A's reported outcome labels, before vs after
  d2 lean change                paired d = b(B) - b(A) on shared events, mean after vs before (Welch z)
  d3 naturality (loop change)   Delta Phi, Phi = mean d(topic 0) - mean d(topic 1), after minus before
  d4 arity-graded monodromy     holonomy of the loop A -> C (after) -> A (before) on item labels, per item-arity
                                parity: Hungarian alignments, moved labels > 0 in the ODD grade
  d5 coverage-graded holonomy   loop A -> B -> C on stance, coverage grade 2, after period (|z| > 1.96)
Also: PROJECTED OUTCOMES at time t: forecast P(o) for late events with a pooled model vs a change-point model
(split at the scanned change point of the outcome series); log loss.
"""
import json, itertools, collections, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
sys.path.insert(0, "../..")
from amb_vigneaux.arity_holonomy import holonomy

K = 3; SRC = ("A", "B", "C")


def generate(rng, plant, n_before=150, n_after=150, delta=1.0):
    pis = {s: rng.permutation(K) for s in SRC}
    u = {"A": 0.0, "B": 0.5, "C": -0.5}; v = [0.0, 0.3, -0.3]
    events = []
    for t in range(n_before + n_after):
        after = t >= n_before
        po = (0.25, 0.6, 0.15) if (after and plant == "A") else (0.6, 0.25, 0.15)
        o = int(rng.choice(K, p=po)); top = int(rng.integers(3))
        items = []
        for _ in range(4):
            k = int(rng.integers(2, 6)); y = o if rng.random() < 0.8 else int(rng.integers(K))
            items.append((k, y))
        cov = {s: rng.random() < 0.8 for s in SRC}
        if sum(cov.values()) == 0:
            cov["A"] = True
        rep = {}
        for s in SRC:
            if not cov[s]:
                continue
            labs = []
            for k, y in items:
                lab = int(pis[s][y])
                if after and plant == "D" and s == "C" and k % 2 == 1:
                    lab = (lab + 1) % K
                if rng.random() < 0.1:
                    lab = int(rng.integers(K))
                labs.append((k, lab))
            b = u[s] + v[top] + rng.normal()
            if after and plant == "B" and s == "B":
                b += delta
            if after and plant == "C" and s == "B" and top == 0:
                b += delta
            if after and plant == "E" and s == "C" and not cov["A"]:
                b += delta
            rep[s] = dict(labels=labs, stance=b, outcome_label=int(pis[s][o]) if rng.random() > 0.1 else int(rng.integers(K)))
        events.append(dict(t=t, after=after, topic=top, o=o, rep=rep))
    return events


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def d1(E):
    tab = np.zeros((2, K))
    for e in E:
        if "A" in e["rep"]:
            tab[int(e["after"]), e["rep"]["A"]["outcome_label"]] += 1
    return chi2_contingency(tab + 0.5)[1] < 0.05


def paired(E, after, topic=None):
    return [e["rep"]["B"]["stance"] - e["rep"]["A"]["stance"] for e in E
            if e["after"] == after and "A" in e["rep"] and "B" in e["rep"] and (topic is None or e["topic"] == topic)]


def d2(E):
    return abs(welch(paired(E, False), paired(E, True))) > 1.96


def d3(E):
    def phi(after):
        a, b = np.array(paired(E, after, 0)), np.array(paired(E, after, 1))
        return a.mean() - b.mean(), a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
    (p0, v0), (p1, v1) = phi(False), phi(True)
    return abs((p1 - p0) / np.sqrt(v0 + v1)) > 1.96


def align(E, after, grade):
    N = np.zeros((K, K))
    for e in E:
        if e["after"] == after and "A" in e["rep"] and "C" in e["rep"]:
            for (k, la), (_, lc) in zip(e["rep"]["A"]["labels"], e["rep"]["C"]["labels"]):
                if k % 2 == grade:
                    N[la, lc] += 1
    r, c = linear_sum_assignment(-N)
    pm = np.empty(K, int); pm[r] = c
    return pm


def d4(E):
    out = {}
    for grade, name in ((1, "odd"), (0, "even")):
        before, after = align(E, False, grade), align(E, True, grade)
        hol = np.array([np.where(before == after[x])[0][0] for x in range(K)])   # before^-1 o after
        out[name] = int((hol != np.arange(K)).sum()) > 0
    return out


def d5(E):
    V = {e["t"]: {s: r["stance"] for s, r in e["rep"].items()} for e in E if e["after"]}
    h = holonomy(V, SRC, 2)
    return h is not None and abs(h["z"]) > 1.96


def project(E, n_test=50):
    """forecast the outcome distribution of the last n_test events (A's labels): pooled vs change-point"""
    lab = [(e["t"], e["rep"]["A"]["outcome_label"]) for e in E if "A" in e["rep"]]
    train = [l for t, l in lab if t < len(E) - n_test]; test = [l for t, l in lab if t >= len(E) - n_test]
    def dist(xs):
        c = np.bincount(xs, minlength=K) + 0.5; return c / c.sum()
    ll = lambda p: -np.mean([np.log(p[x]) for x in test])
    pooled = dist(train)
    # change point by scanning the chi-square statistic over split positions
    best, cut = -1, None
    for s in range(20, len(train) - 20, 5):
        stat = chi2_contingency(np.stack([np.bincount(train[:s], minlength=K), np.bincount(train[s:], minlength=K)]) + 0.5)[0]
        if stat > best:
            best, cut = stat, s
    cp = dist(train[cut:])
    return ll(pooled), ll(cp)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rng = np.random.default_rng(0); R = {"n_per_period": N}
    plants = ["none", "A", "B", "C", "D", "E"]
    names = ["d1 outcomes", "d2 lean", "d3 naturality", "d4 monodromy (odd)", "d4 monodromy (even)", "d5 coverage-graded"]
    reps = 200
    table = {}
    for plant in plants:
        hits = collections.Counter(); pr = []
        for _ in range(reps):
            E = generate(rng, plant, n_before=N, n_after=N)
            hits["d1 outcomes"] += d1(E); hits["d2 lean"] += d2(E); hits["d3 naturality"] += d3(E)
            m = d4(E); hits["d4 monodromy (odd)"] += m["odd"]; hits["d4 monodromy (even)"] += m["even"]
            hits["d5 coverage-graded"] += d5(E)
            if plant in ("none", "A"):
                pr.append(project(E))
        table[plant] = {n: hits[n] / reps for n in names}
        if pr:
            R[f"projection_{plant}"] = dict(pooled=float(np.mean([a for a, _ in pr])), change_point=float(np.mean([b for _, b in pr])))
    R["detection"] = table
    print(f"detection rates (rows: planted phase change; columns: detector), 200 runs each, {N} events per period")
    print(f"{'plant':>6s} | " + " | ".join(f"{n:>19s}" for n in names))
    for plant in plants:
        print(f"{plant:>6s} | " + " | ".join(f"{table[plant][n]:19.2f}" for n in names))
    for k in ("projection_none", "projection_A"):
        print(f"{k}: log loss on the last 50 events  pooled {R[k]['pooled']:.3f}   change-point {R[k]['change_point']:.3f}")
    json.dump(R, open(f"three_layer_dynamics_results_N{N}.json", "w"), indent=1)
