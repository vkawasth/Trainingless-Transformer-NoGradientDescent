"""Checks for the operations ledger LEDGER.yaml (repository root).
  LG1  schema: every row has id, operation, assumptions, parameter_map, levels (contingency, toric, mirror, floer, category), walls (probability,
       support, tropical, critical, stokes, context), wall_certificate and checks; every status is one of the allowed values; ids are unique;
       the category level is never Proved, Exact or Numerical (no categorical claim is made)
  LG2  every check id listed appears as PASS in CHECK_RUN_20261030.log, and none as FAIL
  LG3  the rendered appendix kahler/ledger_table.tex is up to date with LEDGER.yaml
"""
import os, sys, re, yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, "kahler"))
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
L = yaml.safe_load(open(os.path.join(ROOT, "LEDGER.yaml"))); rows = L["rows"]
ALLOWED = {"Proved", "Used", "Cited", "Exact", "Numerical", "Defined", "Open", "N/A"}
LEV = ["contingency", "toric", "mirror", "floer", "category"]; WALLS = ["probability", "support", "tropical", "critical", "stokes", "context"]
ok = len({r["id"] for r in rows}) == len(rows)
for r in rows:
    ok &= all(k in r for k in ("id", "operation", "assumptions", "parameter_map", "levels", "walls", "wall_certificate", "checks"))
    ok &= all(k in r["levels"] and r["levels"][k]["status"] in ALLOWED for k in LEV) and all(k in r["walls"] for k in WALLS)
    ok &= r["levels"]["category"]["status"] not in {"Proved", "Exact", "Numerical"}
check("LG1 ledger schema, allowed statuses, unique ids, no categorical claim", ok, f"{len(rows)} rows")
log = open(os.path.join(ROOT, "CHECK_RUN_20261030.log")).read(); ids = sorted({c for r in rows for c in r["checks"]})
passed = set(re.findall(r"^PASS (\w+)", log, re.M)); failed = set(re.findall(r"^FAIL (\w+)", log, re.M))
missing = [c for c in ids if c not in passed or c in failed]
check("LG2 every cited check is PASS in the run log", not missing, f"{len(ids)} ids; missing {missing}")
import render_ledger
check("LG3 kahler/ledger_table.tex is up to date with LEDGER.yaml", open(os.path.join(ROOT, "kahler", "ledger_table.tex")).read() == render_ledger.render())
print(f"{sum(results)}/{len(results)} passed")
