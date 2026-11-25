"""AMEND3 erratum: rebuild qf5_<p>_<kind>.json from the job log (one line 't winding/pi pass' per selected loop)."""
import sys, json, os
p, kind = int(sys.argv[1]), sys.argv[2]; D = os.path.dirname(os.path.abspath(__file__))
L = [l.split() for l in open(os.path.join(D, f"run_{p}_{kind}.log")) if len(l.split()) == 3 and l.split()[2] in ("True", "False")]
w = [float(x[1]) for x in L]; ok = all(x[2] == "True" for x in L)
json.dump(dict(path=p, kind=kind, w=w, ok=ok, n=len(L), seconds=None, rebuilt_from_log=True), open(os.path.join(D, f"qf5_{p}_{kind}.json"), "w")); print(p, kind, w, ok, len(L))
