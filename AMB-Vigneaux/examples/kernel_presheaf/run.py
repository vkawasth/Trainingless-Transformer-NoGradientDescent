"""Viability check of 'the sheaf of Markov kernels on the cover'.

(1) Acyclic covers: random compatible families on random acyclic covers (chains and stars of contexts,
    arbitrary kernels along the join tree, not built from a global law); one conditional-product sweep
    in a running-intersection order glues them exactly.
(2) Cyclic cover (CHSH 4-cycle), PR box + white noise at weight lambda: along the chain of the first
    three contexts the conditional product always exists and reproduces those three exactly; the
    closing context is off by a TV defect. Compare with CF (LP) and with the existence of a global law.
    The defect is positive even where a global law exists (lambda <= 1/2), so 'a conditional product
    exists' is not the gluing criterion.
(3) Linear descent: every compatible family has a signed global extension (residual ~ 1e-16),
    including the PR box; positivity carries the obstruction.
(4) Descent failure vs gluing failure on real data: the AllSides triangle (layers_chart) families are
    mostly signalling, i.e. they fail descent before gluing is even posed (reported from its results).
"""
import os, sys, json, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.models import pr_box, white_noise
from amb_vigneaux.outcome import contextual_fraction
from amb_vigneaux.functors import extension_exists
from amb_vigneaux.kernel_presheaf import rip_order, sweep_glue, gluing_defects, signed_extension, descent_defect

rng = np.random.default_rng(0)


def random_tree_family(n_vars=6, k=3, arity=2):
    """a random join tree of contexts of size <= arity+1; tables built as parent-overlap marginal x random kernel"""
    names = [f"v{i}" for i in range(n_vars)]
    outs = {v: tuple(range(k)) for v in names}
    ctxs = [tuple(names[:arity + 1])]
    seen = list(ctxs[0])
    while len(seen) < n_vars:
        par = ctxs[int(rng.integers(len(ctxs)))]
        sep = tuple(sorted(rng.choice(par, size=int(rng.integers(1, len(par))), replace=False), key=names.index))
        new = names[len(seen)]; seen.append(new); ctxs.append(sep + (new,))
    sc = Scenario(outs, tuple(ctxs))
    tables = {ctxs[0]: rng.dirichlet(np.ones(sc.n_sections(ctxs[0])) * 0.5)}
    for C in ctxs[1:]:
        sep = C[:-1]; par = next(D for D in tables if set(sep) <= set(D))
        m = sc.restriction_matrix(par, sep) @ tables[par]
        K = rng.dirichlet(np.ones(k) * 0.5, size=len(m))                      # arbitrary kernel sep -> new
        tables[C] = (m[:, None] * K).ravel()
    order = list(ctxs); rng.shuffle(order)
    return EmpiricalModel(Scenario(outs, tuple(order)), {C: tables[C] for C in order})


if __name__ == "__main__":
    out = {}
    # (1)
    worst, n_ok, n = 0.0, 0, 200
    for _ in range(n):
        e = random_tree_family(n_vars=int(rng.integers(4, 8)), k=int(rng.integers(2, 4)))
        order = rip_order(e.scenario.contexts); assert order is not None
        d = max(gluing_defects(e, sweep_glue(e, order)).values()); worst = max(worst, d); n_ok += d < 1e-9
        naive = max(gluing_defects(e, sweep_glue(e, e.scenario.contexts)).values())
    out["acyclic"] = dict(families=n, glued_by_one_sweep=n_ok, worst_defect=worst)
    print(f"(1) acyclic covers: one conditional-product sweep in RIP order glues {n_ok}/{n} random families (worst TV {worst:.1e})")
    # (2)
    PR = pr_box(); sc = PR.scenario; WN = white_noise(sc)
    print("    cyclic order check:", rip_order(sc.contexts))
    chain = list(sc.contexts[:3]); close = sc.contexts[3]
    rows = []
    for lam in np.round(np.linspace(0, 1, 21), 3):
        e = EmpiricalModel(sc, {C: lam * PR.tables[C] + (1 - lam) * WN.tables[C] for C in sc.contexts})
        best = None
        for perm in itertools.permutations(sc.contexts):             # best spanning chain over all orders
            dd = gluing_defects(e, sweep_glue(e, perm)); mx = max(dd.values())
            best = mx if best is None else min(best, mx)
        dd = gluing_defects(e, sweep_glue(e, chain))
        rows.append(dict(lam=float(lam), CF=contextual_fraction(e).value, glues=bool(extension_exists(e)),
                         chain_defect=max(dd[C] for C in chain), closing_defect=dd[close], best_order_defect=best,
                         descent=descent_defect(e), signed_residual=signed_extension(e)[1]))
    print(f"(2) {'lambda':>6s} {'CF':>6s} {'global?':>8s} {'chain':>8s} {'closing':>8s} {'best order':>10s} {'signed res':>10s}")
    for r in rows:
        print(f"    {r['lam']:6.2f} {r['CF']:6.3f} {str(r['glues']):>8s} {r['chain_defect']:8.1e} {r['closing_defect']:8.4f} {r['best_order_defect']:10.4f} {r['signed_residual']:10.1e}")
    out["chsh"] = rows
    # random non-contextual families on the 4-cycle: does one sweep ever glue them?
    hits = []
    for _ in range(200):
        P = rng.dirichlet(np.ones(16) * 0.3); e = EmpiricalModel.from_global(sc, P)
        hits.append(max(gluing_defects(e, sweep_glue(e, sc.contexts)).values()))
    hits = np.array(hits)
    out["nc_cycle"] = dict(n=200, glued=int((hits < 1e-9).sum()), median_closing=float(np.median(hits)))
    print(f"    random NON-contextual families on the 4-cycle: one sweep glues {(hits < 1e-9).sum()}/200 (median closing TV {np.median(hits):.3f});"
          " a global law exists for all 200")
    # (3)
    print(f"(3) PR box: descent defect {descent_defect(PR):.1e}, signed-extension residual {signed_extension(PR)[1]:.1e}, CF {contextual_fraction(PR).value:.3f}")
    # (4)
    lc = os.path.join(here, "../outcomes/layers_chart_results.json")
    if os.path.exists(lc):
        L = json.load(open(lc)); sig = np.array([r["signalling"] for r in L]); cf = np.array([r["CF"] for r in L])
        out["allsides_triangle"] = dict(windows=len(L), median_signalling=float(np.median(sig)), median_CF=float(np.median(cf)),
                                        corr=float(np.corrcoef(sig, cf)[0, 1]))
        print(f"(4) AllSides triangle: {len(L)} windows, median signalling (descent defect) {np.median(sig):.3f}, median CF {np.median(cf):.3f},"
              f" corr {np.corrcoef(sig, cf)[0, 1]:.2f}: the families mostly fail descent, not gluing")
    json.dump(out, open(os.path.join(here, "results.json"), "w"), indent=1)
