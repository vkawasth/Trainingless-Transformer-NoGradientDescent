# Bands pilot (NHANES + COVID-19 Forecast Hub)

Run from this folder: `python3 nhanes_pilot.py`, `python3 fhub_fetch.py` (needs the partial clone in data/fhub), `python3 fhub_pilot.py`.

Data sources
- NHANES 2009-10 and 2011-12 (NHANESraw from the CRAN package NHANES, cloned from github.com/cran/NHANES), adults 20+, MEC weights,
  stratified PSU jackknife for sampling error.
- reichlab/covid19-forecast-hub: COVIDhub-ensemble state-level weekly incident-death quantile forecasts (forecast dates
  2021-06-07..2022-04-04) and the truth-file vintages as they stood on each forecast date; final truth = vintage 2022-05-09.

Outputs (out/)
- nhanes_summary.json, nhanes_cells.csv (16 subgroups: risk, sampling band, LP bands at each information level),
  nhanes_revision.csv (2009-10 vs 2011-12), nhanes_simpson.csv
- fhub_summary.json, fhub_bands.csv (every band with wall, side, margin, speed, tau), fhub_backfill.csv
