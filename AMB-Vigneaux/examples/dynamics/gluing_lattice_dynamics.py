"""Gluing-lattice dynamics: which invariants jump and which move continuously along a path of models.

Path: e(lambda) = lambda * PR box + (1 - lambda) * white noise on the CHSH scenario (4 contexts, a 4-cycle).
Along lambda we track
  CF        contextual fraction (LP)                              probability layer, continuous
  S0        mean log support size over contexts (tolerance tau)  support layer, jumps
  gamma     AMB cohomological obstruction on the support at tolerance tau (sound, not complete)
  S2        mean Tsallis-2 entropy over contexts                  continuous
  IPF       maximum-entropy extension converges? (converges iff a joint exists, i.e. CF = 0)
  lattice   statistical verdict from N samples per context: the full cover is 'obstructed' if the CHSH value
            exceeds 2 by 1.645 se (one-sided 5%); every proper sub-cover (3 of the 4 contexts) is a tree and
            always glues (Vorob'ev), so the full cycle is the only possible minimal obstruction.
The support tweak: an empirical support uses a tolerance tau (cells with p < tau count as impossible). With
tau = 0.02 the support of e(lambda) collapses to the PR support at lambda > 1 - 4 tau = 0.92.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux.models import pr_box, white_noise
from amb_vigneaux.scenario import EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction, analyse_outcomes
from amb_vigneaux.functors import max_entropy_extension

TAU = 0.02
rng = np.random.default_rng(0)
PR = pr_box(); SC = PR.scenario; WN = white_noise(SC)


def mix(lam):
    return EmpiricalModel(SC, {C: lam * PR.tables[C] + (1 - lam) * WN.tables[C] for C in SC.contexts})


def chsh_value(tables):
    """S = sum over contexts of +-(P(a=b) - P(a!=b)) with the PR sign pattern (x*y = 1 flips the sign)"""
    S = 0.0
    for C in SC.contexts:
        secs = SC.sections(C); t = tables[C]
        corr = sum(p * (1 if s[0] == s[1] else -1) for s, p in zip(secs, t))
        x, y = int(C[0][1]), int(C[1][1])
        S += -corr if (x & y) else corr
    return S


def obstructed(model, N):
    counts = {C: rng.multinomial(N, model.tables[C]) for C in SC.contexts}
    t = {C: c / N for C, c in counts.items()}
    S = chsh_value(t)
    var = 0.0
    for C in SC.contexts:
        secs = SC.sections(C); pr = t[C]
        e = np.array([1 if s[0] == s[1] else -1 for s in secs], float)
        m = float((pr * e).sum()); var += (1 - m * m) / N
    return S - 2 > 1.645 * np.sqrt(var)


if __name__ == "__main__":
    lams = np.round(np.concatenate([np.linspace(0, 0.48, 13), [0.52], np.linspace(0.56, 1.0, 23)]), 4)
    rows = []
    for lam in lams:
        e = mix(lam)
        cf = contextual_fraction(e).value
        rep = analyse_outcomes(e, eps=TAU, with_cf=False)
        supp = e.support(TAU)
        S0 = float(np.mean([np.log(len(supp[C])) for C in SC.contexts]))
        S2 = float(np.mean([1 - (e.tables[C] ** 2).sum() for C in SC.contexts]))
        me = max_entropy_extension(e, iters=3000, tol=1e-8)
        det = {N: float(np.mean([obstructed(e, N) for _ in range(300)])) for N in (50, 200, 1000, 5000)}
        rows.append(dict(lam=float(lam), CF=cf, S0=S0, S2=S2, gamma=bool(rep.gamma_h1_nonzero), level=rep.level(),
                         strong=bool(rep.strongly_contextual), ipf_converged=bool(me.converged), ipf_residual=float(me.residual),
                         chsh=chsh_value(e.tables), detect=det))
    print(f"{'lambda':>6s} {'CF':>6s} {'CHSH':>6s} {'S0':>6s} {'S2':>6s} {'gamma':>6s} {'IPF':>9s}  detection at N = 50 / 200 / 1000 / 5000")
    for r in rows:
        print(f"{r['lam']:6.3f} {r['CF']:6.3f} {r['chsh']:6.3f} {r['S0']:6.3f} {r['S2']:6.3f} {str(r['gamma']):>6s} "
              f"{('conv' if r['ipf_converged'] else 'cycles'):>9s}  " + " / ".join(f"{r['detect'][N]:.2f}" for N in (50, 200, 1000, 5000)))
    # regimes
    cfpos = [r for r in rows if r["CF"] > 1e-9]
    stretch = [r["lam"] for r in cfpos if not r["gamma"]]
    print(f"\nCF > 0 from lambda = {min(r['lam'] for r in cfpos):.2f};  CF > 0 but gamma = 0 on lambda in [{min(stretch):.2f}, {max(stretch):.2f}];"
          f"  gamma != 0 from lambda = {min(r['lam'] for r in rows if r['gamma']):.2f}")
    print(f"IPF converges exactly where CF = 0: {all(r['ipf_converged'] == (r['CF'] < 1e-9) for r in rows)}")
    lin = np.polyfit([r['lam'] for r in cfpos], [r['CF'] for r in cfpos], 1)
    print(f"CF on the contextual side is linear in lambda: CF = {lin[0]:.3f} lambda {lin[1]:+.3f}")
    json.dump(rows, open(os.path.join(here, "gluing_lattice_dynamics_results.json"), "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted = "#0b0b0b", "#52514e"; cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    L = [r["lam"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for a in ax.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=8); a.grid(axis="y", color="#e6e5e1", lw=0.6)
        a.axvline(0.5, color=muted, lw=0.8, ls=":"); a.axvline(1 - 4 * TAU, color=muted, lw=0.8, ls="--")
    ax[0, 0].plot(L, [r["CF"] for r in rows], color=cols[0], lw=2); ax[0, 0].set_title("CF (probability layer)", fontsize=9, loc="left", color=ink)
    ax[0, 1].step(L, [r["S0"] for r in rows], where="mid", color=cols[0], lw=2)
    ax[0, 1].set_title("S0 = mean log support size (support layer)", fontsize=9, loc="left", color=ink)
    for r0, r1 in zip(rows[:-1], rows[1:]):
        if r0["gamma"]:
            ax[0, 1].axvspan(r0["lam"], r1["lam"], color=cols[1], alpha=0.15, lw=0)
    ax[0, 1].text(0.955, 1.2, "γ ≠ 0", color=cols[1], fontsize=8, ha="center", rotation=90)
    ax[1, 0].semilogy(L, [max(r["ipf_residual"], 1e-9) for r in rows], color=cols[0], lw=2)
    ax[1, 0].set_title("IPF residual after 3000 sweeps", fontsize=9, loc="left", color=ink)
    for k, N in enumerate((50, 200, 1000, 5000)):
        ax[1, 1].plot(L, [r["detect"][N] for r in rows], color=cols[k], lw=2, label=f"N = {N}")
    ax[1, 1].set_title("statistical verdict: P(obstructed)", fontsize=9, loc="left", color=ink)
    ax[1, 1].legend(fontsize=7, frameon=False, loc="upper left")
    for a in ax[1]:
        a.set_xlabel("λ (weight of the PR box)", fontsize=8, color=muted)
    fig.tight_layout(); fig.savefig(os.path.join(here, "gluing_lattice_dynamics.pdf")); fig.savefig(os.path.join(here, "gluing_lattice_dynamics.png"), dpi=150)
