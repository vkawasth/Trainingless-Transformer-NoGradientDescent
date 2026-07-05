"""
anchor_probe.py
================
Implements the Anchor structure discussed for the DP compiler's backtracking
system:

    Anchor = {
        param_value: float     # e.g. LR multiplier, step count, or alpha
                                # along a fixed direction
        state_key: tuple       # (val_bucket, phi, tau_bucket, phase_bucket)
        classification: 'good' | 'bad' | 'boundary'
        metrics: {val, phi, tau, rm2sigma}
    }

Design:
  - An anchor comes from a CHEAP probe (a handful of steps or a single
    forward-eval along a fixed direction), not a full run to convergence.
  - Forward exploration uses BISECTION between a known-good and known-bad
    param_value, rather than a fixed hand-picked grid/sequence -- this is
    what makes it corpus/architecture independent instead of hardcoding
    LR values like 10, 3, 0.003.
  - Backtracking reuses anchors already collected at a given state_key
    instead of re-probing blindly.
  - Once enough classified anchors exist, refine the estimate of the best
    param_value with a guarded polynomial fit (quadratic by default, cubic
    if enough points AND the fit's minimum lands inside the tested range --
    never extrapolated, since an untested region has no support for the fit).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Anchor:
    param_value: float
    state_key: tuple
    classification: str  # 'good' | 'bad' | 'boundary'
    metrics: dict = field(default_factory=dict)  # {val, phi, tau, rm2sigma}


def classify(metrics, phi_target=4, tau_range=(5.0, 7.5), rm2_min=0.65,
             val_explode_mult=1.5, val_ref=None):
    """
    Classify a probe's metrics as good/bad/boundary using the SAME geometric
    criteria already used elsewhere in this codebase (phi_clean, tau band,
    rm2sigma), rather than inventing a new one.
    """
    val, phi, tau, rm2 = metrics['val'], metrics['phi'], metrics['tau'], metrics.get('rm2sigma', 0.0)

    if val_ref is not None and val > val_ref * val_explode_mult:
        return 'bad'  # blew up -- unconditionally bad regardless of geometry

    geo_good = (phi >= phi_target and tau_range[0] <= tau <= tau_range[1] and rm2 >= rm2_min)
    if geo_good:
        return 'good'

    # "boundary": geometry is partially there (close to target on at least
    # one axis) -- useful for bisection to know it's near the transition,
    # not just uniformly bad.
    near_phi = phi >= phi_target - 1
    near_tau = (tau_range[0] - 1.5) <= tau <= (tau_range[1] + 1.5)
    if near_phi and near_tau:
        return 'boundary'

    return 'bad'


class AnchorRegistry:
    """Per-state_key collection of anchors, supporting bisection search and
    reuse across backtracking."""

    def __init__(self):
        self._anchors = {}  # state_key -> list[Anchor]

    def add(self, anchor: Anchor):
        self._anchors.setdefault(anchor.state_key, []).append(anchor)

    def get(self, state_key):
        return self._anchors.get(state_key, [])

    def best_anchor(self, state_key):
        """Anchor with lowest val among 'good' anchors at this state, else
        the anchor closest to boundary (smallest geometric distance)."""
        anchors = self.get(state_key)
        if not anchors:
            return None
        good = [a for a in anchors if a.classification == 'good']
        if good:
            return min(good, key=lambda a: a.metrics['val'])
        return min(anchors, key=lambda a: a.metrics['val'])

    def bisect_probe(self, state_key, probe_fn, lo, hi, max_probes=6,
                      val_ref=None, n_grid=3):
        """
        Search for a good param_value in [lo, hi].

        IMPORTANT: val vs. param_value is typically U-shaped (too small a
        step/LR = slow, bad geometry; too large = overshoot, bad geometry),
        not monotonic with a single good/bad transition. Plain two-endpoint
        bisection (assuming lo is bad, hi is good or vice versa) fails on a
        U-shaped response with both endpoints bad and the good region
        strictly inside -- this was caught by a synthetic test before this
        was wired into real code.

        Approach: coarse log-spaced grid first (n_grid points, cheap probes)
        to locate which sub-interval contains the best response, then a
        ternary-search-style narrowing within that sub-interval for the
        remaining probe budget. Works for both the monotonic case (grid
        will show a clear best-at-one-end) and the interior-bracket case.

        probe_fn(param_value) -> metrics dict {val, phi, tau, rm2sigma}

        Returns the list of Anchors created during this call.
        """
        created = []

        def probe_and_record(pv):
            metrics = probe_fn(pv)
            cls = classify(metrics, val_ref=val_ref)
            a = Anchor(pv, state_key, cls, metrics)
            self.add(a)
            created.append(a)
            return a

        use_log = lo > 0 and hi > 0 and (hi / max(lo, 1e-8)) > 3.0
        if use_log:
            grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
        else:
            grid = np.linspace(lo, hi, n_grid)

        grid_anchors = [probe_and_record(float(pv)) for pv in grid]
        remaining = max_probes - n_grid

        # Locate the best grid point and narrow to its neighboring interval.
        best_i = int(np.argmin([a.metrics['val'] for a in grid_anchors]))
        left_i = max(best_i - 1, 0)
        right_i = min(best_i + 1, len(grid) - 1)
        cur_lo, cur_hi = float(grid[left_i]), float(grid[right_i])

        # Ternary-search-style narrowing within the located bracket.
        for _ in range(remaining):
            if cur_hi <= cur_lo:
                break
            if use_log:
                m1 = np.exp(np.log(cur_lo) + (np.log(cur_hi) - np.log(cur_lo)) / 3)
                m2 = np.exp(np.log(cur_lo) + 2 * (np.log(cur_hi) - np.log(cur_lo)) / 3)
            else:
                m1 = cur_lo + (cur_hi - cur_lo) / 3
                m2 = cur_lo + 2 * (cur_hi - cur_lo) / 3
            a1 = probe_and_record(float(m1))
            a2 = probe_and_record(float(m2))
            if a1.metrics['val'] <= a2.metrics['val']:
                cur_hi = float(m2)
            else:
                cur_lo = float(m1)

        return created

    def refine_with_polynomial(self, state_key, prefer_cubic=True):
        """
        Guarded polynomial refinement: fit val as a function of param_value
        (in log-space if all param_values are positive and span >1 order of
        magnitude, since LR-like quantities are naturally log-scaled) using
        the anchors collected so far at this state_key, and return the
        param_value that minimizes the fit.

        Guard: the fit's minimum is only trusted if it falls INSIDE the
        range of param_values actually tested. A polynomial fit minimizing
        outside the tested range is extrapolation with no support from data
        and is rejected -- falls back to the best directly-observed anchor.

        Returns (param_value, degree_used) or (None, None) if too few points.
        """
        anchors = self.get(state_key)
        if len(anchors) < 3:
            return None, None

        pvs = np.array([a.param_value for a in anchors], dtype=float)
        vals = np.array([a.metrics['val'] for a in anchors], dtype=float)

        use_log = np.all(pvs > 0) and (pvs.max() / max(pvs.min(), 1e-8)) > 3.0
        x = np.log(pvs) if use_log else pvs

        degree = 3 if (prefer_cubic and len(anchors) >= 4) else 2
        try:
            coeffs = np.polyfit(x, vals, degree)
        except np.linalg.LinAlgError:
            return None, None

        # Find critical points of the fitted polynomial inside [x.min(), x.max()]
        deriv = np.polyder(coeffs)
        roots = np.roots(deriv)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
        candidates = [r for r in real_roots if x.min() <= r <= x.max()]

        if not candidates:
            return None, None  # fit's minimum isn't supported by tested data

        # Pick the candidate with the lowest fitted value
        fitted_vals = [np.polyval(coeffs, r) for r in candidates]
        best_r = candidates[int(np.argmin(fitted_vals))]
        best_pv = float(np.exp(best_r)) if use_log else float(best_r)
        return best_pv, degree
