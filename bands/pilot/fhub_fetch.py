"""Extract state-level weekly incident-death forecasts (COVIDhub-ensemble) and weekly truth vintages from a
blob-less partial clone of reichlab/covid19-forecast-hub. Only the needed rows are kept on disk."""
import io, subprocess, sys, datetime as dt
import pandas as pd

REPO = "data/fhub"
def git(*a, text=True):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=text, check=True).stdout

def state_rows(csv_text, is_forecast):
    keep = []
    lines = csv_text.splitlines(); header = lines[0]
    for ln in lines[1:]:
        if is_forecast and "inc death" not in ln: continue
        keep.append(ln)
    df = pd.read_csv(io.StringIO(header + "\n" + "\n".join(keep)), dtype={"location": str})
    if "location" not in df.columns: return None
    return df[df.location.str.len() == 2]

# forecasts: Delta and Omicron waves
dates = pd.date_range("2021-06-07", "2022-04-04", freq="7D")
import os
fc = []
for d in ([] if os.path.exists("data/fc_state_incdeath.csv") else dates):
    path = f"data-processed/COVIDhub-ensemble/{d.date()}-COVIDhub-ensemble.csv"
    try:
        txt = git("show", f"HEAD:{path}")
    except subprocess.CalledProcessError:
        print("missing", path, file=sys.stderr); continue
    f = state_rows(txt, True); f = f[f.type == "quantile"]
    fc.append(f); print(d.date(), len(f), file=sys.stderr)
if fc: pd.concat(fc).to_csv("data/fc_state_incdeath.csv", index=False)

# truth vintages: the last commit of the truth file on or before each forecast date + 1 day (data available at forecast time)
log = git("log", "--format=%H %cI", "--", "data-truth/truth-Incident Deaths.csv").split("\n")
commits = [(l.split()[0], pd.Timestamp(l.split()[1]).tz_convert(None)) for l in log if l.strip()]
vint = []
for d in list(dates) + list(pd.date_range(dates[-1] + pd.Timedelta(days=7), "2023-03-06", freq="28D")):
    cands = [c for c in commits if c[1] <= d + pd.Timedelta(days=1)]
    if not cands: continue
    h, t = max(cands, key=lambda c: c[1])
    try:
        txt = git("show", f"{h}:data-truth/truth-Incident Deaths.csv")
    except subprocess.CalledProcessError:
        continue
    v = state_rows(txt, False)
    if v is None: print("skip", d.date(), file=sys.stderr); continue
    v["vintage"] = d.date(); v["commit_time"] = t
    vint.append(v); print("vintage", d.date(), h[:9], len(v), file=sys.stderr)
pd.concat(vint).to_csv("data/truth_vintages_incdeath.csv", index=False)
