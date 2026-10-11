# MIMIC-IV run of the structural-change monitor (Phase II)

The script runs entirely on your machine. It reads MIMIC-IV, keeps patient-level data in memory only, and writes **aggregate** results
(`results/results.json`, `results/results.txt`). The analysis is fixed in `PREANALYSIS_PLAN.md`, dated before any data are seen.

## 1. Get access (one time)
1. Create an account at https://physionet.org and apply for **credentialed access** (profile, then *Credentialing*).
2. Complete the CITI course **"Data or Specimens Only Research"** and upload the certificate (profile, then *Training*).
3. When approved, open https://physionet.org/content/mimiciv/ (current version **3.1**) and **sign the data use agreement**.

## 2. Download only the files needed (about 10 GB compressed, mostly `chartevents`)
```bash
U=your_physionet_username
B=https://physionet.org/files/mimiciv/3.1
for f in hosp/patients hosp/admissions hosp/labevents hosp/d_labitems \
         icu/icustays icu/chartevents icu/inputevents icu/procedureevents icu/d_items; do
  wget -N -c -x -nH --cut-dirs=3 --user "$U" --ask-password "$B/$f.csv.gz"
done
```
The files land in `./hosp/` and `./icu/` under the current folder. Keep them compressed; the script reads `.csv.gz` directly.
If the version on the project page is newer than 3.1, change `B` and note the version in your write-up.

## 3. Install and test (no MIMIC data needed)
```bash
python3 -m venv .venv && source .venv/bin/activate        # optional
pip install -r requirements.txt
python3 check_synthetic.py           # builds fake MIMIC-shaped data with planted answers and checks the pipeline: 8/8 PASS
```

## 4. Run on MIMIC-IV
```bash
python3 mimic_monitor.py /path/to/folder/containing/hosp_and_icu --out results --threads 8
```
- It first checks every item identifier against `d_items` / `d_labitems` and stops if a label does not match.
- Expect roughly 10-40 minutes and several GB of RAM (DuckDB scans `chartevents` once); `--quick` gives a fast, low-resolution run.
- Output: `results/results.json`. Proportion tables with any cell below 10 are suppressed (`--min-cell` to raise it).

## 5. Sharing
- Do not upload the MIMIC files, or anything with patient-level rows, to any online service, including AI assistants
  (PhysioNet's responsible-use guidance: https://physionet.org/news/post/llm-responsible-use/).
- `results.json` holds only aggregates. Read it yourself and check it against the data use agreement before sharing it.
- Once it is cleared, it can be written up as *Empirical Site Separation II*.

## Files
| file | what it does |
|---|---|
| `PREANALYSIS_PLAN.md` | definitions, thresholds, analyses A-E and decision rules, fixed in advance |
| `mimic_monitor.py` | extraction (DuckDB) and analyses A-E; aggregate output only |
| `make_synthetic_mimic.py`, `check_synthetic.py` | synthetic data in MIMIC-IV layout with planted answers, and the end-to-end test |
| `monitor.py`, `cbd.py`, `panel_models.py`, `panel_paths.py` | the library of the validation paper (copied from `monitor/` and `bridges/cbd/`) |

## Update 2026-10-10
- Analysis E now reports restricted selection models with model-specific nulls (see the amendment at the end of `PREANALYSIS_PLAN.md`). The synthetic check has 9 items.
- `requirements.txt` is unchanged: the logistic fit is done in numpy.
