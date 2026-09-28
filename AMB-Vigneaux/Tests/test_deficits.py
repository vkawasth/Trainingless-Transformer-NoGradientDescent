import numpy as np
from amb_vigneaux import models
from amb_vigneaux.deficits import renyi_deficit


def profile(m, alphas=(0, 0.5, 1)):
    return [renyi_deficit(m, a, iters=3000) for a in alphas]


def test_dial_zero_sets_and_monotonicity():
    pr, hardy, ts, nc = models.pr_box(), models.hardy_model(), models.tsirelson_box(), models.noisy(models.pr_box(), 0.4)
    for m in (pr, hardy, ts, nc):
        d = profile(m)
        assert d[0] <= d[1] + 1e-6 <= d[2] + 2e-6                      # non-decreasing in alpha
    assert profile(pr)[0] > 0.28                                          # strong: Delta_0 > 0
    for m in (hardy, ts):
        d = profile(m)
        assert d[0] < 1e-6 and d[1] > 1e-3 and d[2] > 1e-3               # contextual, not strong
    assert max(profile(nc)) < 1e-6                                        # non-contextual: all zero


def test_delta0_lower_bound_for_strong_contextuality():
    # Jensen + logical Bell: Delta_0 >= log(n / (n - 1)) for n contexts, equality for the PR box
    assert abs(renyi_deficit(models.pr_box(), 0, iters=3000) - np.log(4 / 3)) < 1e-4


def test_repair_hints_follow_trust_and_undo_the_tamper():
    from amb_vigneaux.deficits import repair_hints
    from amb_vigneaux.channels import noisy_shift, edge_family, cycle
    p, G = 3, cycle(4)
    m = edge_family(G, [noisy_shift(p, x, 0.2) for x in [0, 0, 0, 1]])
    ctx = m.scenario.contexts
    h, _ = repair_hints(m)
    costs = [h[c]["cost"] for c in ctx]
    assert max(costs) - min(costs) < 1e-6                           # equal trust: blame is the whole cycle
    for bad in (1, 3):                                               # blame follows the declared trust
        h, _ = repair_hints(m, {c: (20.0 if c != ctx[bad] else 1.0) for c in ctx})
        assert max(ctx, key=lambda c: h[c]["cost"]) == ctx[bad]
    h, _ = repair_hints(m, {c: (20.0 if c != ctx[3] else 1.0) for c in ctx})
    d = h[ctx[3]]["delta"].reshape(p, p)
    assert all(d[x, x] > 0.15 and d[x, (x + 1) % p] < -0.15 for x in range(p))   # undo the +1 shift


def test_grid_hint_names_the_changed_residue():
    from amb_vigneaux.deficits import repair_hints
    gm = models.parity_grid((0, 0, 0), (0, 0, 1), 3); ctx = gm.scenario.contexts
    h, _ = repair_hints(gm, {c: (20.0 if c != ctx[5] else 1.0) for c in ctx})
    agg = {r: 0.0 for r in range(3)}
    for s, x in zip(h[ctx[5]]["sections"], h[ctx[5]]["delta"]):
        agg[sum(s) % 3] += x
    assert agg[0] > 0.9 and agg[1] < -0.9
