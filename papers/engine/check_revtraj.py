"""Checks for the general form of Prop. ordrev and the trajectory comparison (companion note, 'Ordered co-revision'; modules
pce/ordrev.py, pce/revtraj.py, pce/tiebraid.py). Exact enumeration of family F (four sources on a line, pairs 1, 2, 3): 720 orderings of
one adoption and one withdrawal per pair, and 90 orderings of two adoptions per pair.
  RV1 IDENTITY  the partially commutative signature (variables of independent pairs commute) induces exactly the braid partition, from
                level 3 on; at level 2 it is coarser; the ordinary signature is finer (prop:ordrev, general form)
  RV2 FINDING   trajectories in which revisions move reports along the segment joining a pair ('partial') or merge and then retrace
                ('retrace') give the trivial braid for all 720 histories: they carry no order information
  RV3 FINDING   trajectories in which a merged pair separates sideways give 13 classes, most of them with several resolutions; the event
                braid is among a trajectory's resolutions only when both are trivial, and neither the event braid nor the trajectory class
                determines the other: the event braid is a compression of order, not the topology of the trajectories
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import ordrev as OR, revtraj as RT
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def part(H, f):
    c = {}
    for h in H: c.setdefault(f(h), []).append(h)
    return c
def refines(a, b):
    lab = {h: i for i, v in enumerate(b.values()) for h in v}; return all(len({lab[h] for h in v}) == 1 for v in a.values())
def same(a, b): return refines(a, b) and refines(b, a)
F = OR.arrangements({(1, 1): 1, (1, -1): 1, (2, 1): 1, (2, -1): 1, (3, 1): 1, (3, -1): 1})
Fp = OR.arrangements({(1, 1): 2, (2, 1): 2, (3, 1): 2})

# RV1
ok = True; info = []
for nm, H in (("signed", F), ("positive", Fp)):
    br = part(H, lambda h: OR.braid_key(h, 4)); sg = part(H, lambda h: OR.sig_key(h, 3, 6))
    pcs = {lv: part(H, lambda h, lv=lv: OR.pc_sig_key(h, 3, lv)) for lv in (2, 3, 4, 6)}
    ok &= all(same(pcs[lv], br) for lv in (3, 4, 6)) and not same(pcs[2], br) and refines(sg, br) and len(sg) > len(br)
    info.append(f"{nm}: {len(H)} histories, braid {len(br)}, partially commutative signature levels 2/3/4/6: {[len(pcs[l]) for l in (2, 3, 4, 6)]}, ordinary signature {len(sg)}")
check("RV1 IDENTITY partially commutative signature = braid (from level 3); the ordinary signature over-separates", ok, "; ".join(info))

# RV2, RV3
ident = OR.braid_key((), 4); summ = {}
for mode in ("partial", "retrace", "sideways"):
    triv = det = inc = inc_nt = 0; by_traj = {}; by_ev = {}
    for h in F:
        ks, nw = RT.braids(h, mode, n=24); ev = OR.braid_key(h, 4)
        triv += ks == {ident}; det += len(ks) == 1; inc += ev in ks; inc_nt += (ev in ks and ev != ident)
        by_traj.setdefault(frozenset(ks), set()).add(ev); by_ev.setdefault(ev, set()).add(frozenset(ks))
    summ[mode] = dict(trivial=triv, determined=det, event_in_resolutions=inc, event_in_resolutions_nontrivial=inc_nt, classes=len(by_traj),
                      traj_determines_event=all(len(v) == 1 for v in by_traj.values()), event_determines_traj=all(len(v) == 1 for v in by_ev.values()))
json.dump(summ, open(os.path.join(HERE, "revtraj_summary.json"), "w"), indent=1)
check("RV2 FINDING partial and retracing revision: every trajectory braid is trivial (no order information)",
      all(summ[m]["trivial"] == 720 and summ[m]["classes"] == 1 for m in ("partial", "retrace")), f"{ {m: summ[m] for m in ('partial', 'retrace')} }")
s = summ["sideways"]; nev_triv = sum(OR.braid_key(h, 4) == ident for h in F)
check("RV3 FINDING sideways separation: the event braid matches the trajectory only when both are trivial; neither determines the other",
      s["event_in_resolutions_nontrivial"] == 0 and s["event_in_resolutions"] == nev_triv and not s["traj_determines_event"] and not s["event_determines_traj"] and s["classes"] > 1,
      f"{s}; event braids that are trivial: {nev_triv}")
print(f"{sum(results)}/{len(results)} passed")
