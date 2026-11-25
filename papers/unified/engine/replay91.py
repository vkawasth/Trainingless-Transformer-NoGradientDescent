"""EXPLORATORY (not a registered check): replay the Study 91 event lists as streaming data. Events (collisions, caustics) are letters on
strand pairs; two events commute when their strand pairs are disjoint (far commutation; cor:anycollision for collisions). With a delay
window w, an event is committed once w has passed with no later event sharing a strand inside the window; otherwise it stays pending.
Records pending load and the fraction of order-sensitive overlaps (pairs within w sharing a strand)."""
import os, glob, pickle, json, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "study91_3x4", "walks"); out = {}
for f in sorted(glob.glob(os.path.join(D, "walk_9102_*.pkl"))):
    pk = pickle.load(open(f, "rb")); ev = sorted([e for e in pk["ev"] if e["kind"] in ("collision", "caustic")], key=lambda e: e["t"])
    ts = np.array([e["t"] for e in ev]); pairs = [set(int(x) for x in e["pair"]) for e in ev]; res = {}
    for w in (0.025, 0.1, 0.25, 1.0):
        sens = 0; tot = 0; pend = []
        for i in range(len(ev)):
            j = i + 1
            while j < len(ev) and ts[j] - ts[i] <= w:
                tot += 1; sens += bool(pairs[i] & pairs[j]); j += 1
        for t in np.arange(0, 13, 0.05):                       # pending at time t: events in (t - w, t] sharing a strand with another event within w
            idx = np.where((ts > t - w) & (ts <= t))[0]
            pend.append(sum(any(pairs[a] & pairs[b] for b in idx if b != a) for a in idx))
        res[str(w)] = dict(pairs_within_w=tot, order_sensitive=sens, frac=round(sens / tot, 3) if tot else None, mean_pending=round(float(np.mean(pend)), 2), max_pending=int(max(pend)))
    out[os.path.basename(f)] = dict(events=len(ev), **res)
json.dump(out, open("replay91_output.json", "w"), indent=1)
for k, v in out.items(): print(k, v["events"], {w: (v[w]["frac"], v[w]["mean_pending"], v[w]["max_pending"]) for w in ("0.025", "0.1", "0.25", "1.0")})
