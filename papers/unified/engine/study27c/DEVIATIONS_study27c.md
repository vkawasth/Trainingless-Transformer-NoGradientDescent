# Study 27c — deviations from the frozen specification (SHA-256 21632b08...)
None. Code test before the run: SMOKE27C=1 (seeds offset by 500, 30 runs per scenario, output to /dev/null), used only for crashes.
Observation (not a deviation): in both configurations the calibrated z is the bottom of the frozen grid (2.00); the honest calibration
rate there was already 0.016 (A) and 0.033 (B). A lower z might have bought sensitivity, but the grid was fixed and was not extended.
