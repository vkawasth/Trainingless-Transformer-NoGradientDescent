# Phase II on open clinical data

Pre-registered in `PREANALYSIS_AMENDMENT_OPEN_DATA.md` (2026-10-10), which extends the MIMIC-IV plan of 2026-10-09.
- `synth/check_synth.py`: the adapters recover planted answers on synthetic data in each layout (8/8).
- `challenge2012.py a`, `diabetes130.py` and `heart.py` produced the results in `results/frozen/`, frozen with SHA-256 in `results/FROZEN.txt` before set B was opened.
- `challenge2012.py b` and `challenge2012.py compare` give the replication and the negative control (`results/results_2012_compare.json`).
- Data: put `set-a.zip`, `set-b.zip`, `Outcomes-a.txt` and `Outcomes-b.txt` in `data/challenge2012/`, and `diabetes130.zip` and `heartdisease.zip` in `data/uci/`. All are open licences: ODC-By for the Challenge 2012 files, CC BY 4.0 for the UCI files.

## Tower, Fano tubes and ICU interchange (2026-10-10)
- **Plan:** `PREANALYSIS_TOWER_ICU.md`.
- **Synthetic adapter checks:** `python3 tower_icu.py synth`. Output is saved in `results/synth_tower_icu.txt`.
- **Run:**
  1. `python3 tower_icu.py load` builds a patient-level cache under `$PHYSIONET_DIR/cache`. It is kept outside this folder and must never be shared.
  2. `python3 tower_icu.py tower` and `python3 tower_icu.py interchange` run the analyses.
- **Hashes:** `results/FROZEN_TOWER_ICU.txt`.
- **Exploratory, not pre-registered:** `tower_icu_explore.py`, which covers:
  - the level-one obstruction certificate;
  - whether each observed chamber is decided on its bands;
  - the evidence-floor repair of the reliability level.
- **Figure:** `../../make_fig_fano.py` draws `figs/fig_fano_tubes.pdf`.
