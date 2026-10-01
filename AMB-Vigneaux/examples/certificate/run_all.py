"""Robustness certificate on BASIL, MBIC, NewsWCL50 and AllSides (see amb_vigneaux/certificate.py).

Data (not redistributed), via environment variables as in examples/bias3 and examples/allsides:
  BASIL_DIR, MBIC_XLSX, NEWSWCL50_CSV, ALLSIDES_DIR.
For each dataset: Hodge split on the source x topic graph; Fiedler gap vs bootstrap ||dL|| (Weyl margin);
obstruction Q ~ chi2(beta_1), SNR and cosine stability of the harmonic part; survival under random refinement of
the topic cover; leave-one-source-out.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), os.path.join(here, "../allsides")]
import numpy as np
from amb_vigneaux.certificate import certificate


def basil(exclude_trump=False):
    import bias3_spans
    bias3_spans.EXCLUDE.clear()
    if exclude_trump:
        bias3_spans.EXCLUDE.update({"Donald_Trump", "Trump", "Donald_Trump_Jr."})
    return [(s, c, float(pol == "pos"), e) for e, s, w, th, c, pol in bias3_spans.load_spans()]


def mbic():
    import mbic as M
    d = M.load()
    s = d.groupby("sentence_id").agg(y=("b", "mean"), outlet=("outlet", "first"), topic=("topic", "first"))
    return [(o, t, float(v), i) for i, (v, o, t) in zip(s.index, s[["y", "outlet", "topic"]].itertuples(index=False))]


def newswcl50(drop=()):
    import newswcl50 as N
    p = N.load(); p = p[p.tgt.isin(N.TOP) & ~p.tgt.isin(drop)]
    return [(o, t, float(v), e) for e, o, t, v in p[["event_id", "publisher_id", "tgt", "pos"]].itertuples(index=False)]


def allsides(key="outcome"):
    import load
    return [(r["side"], r["group"], float(r[key]), (r["date"], r["topic"])) for r in load.load() if r["group"] != "other"]


if __name__ == "__main__":
    DS = {"BASIL spans (source x target party)": lambda: basil(False),
          "BASIL spans, Trump excluded": lambda: basil(True),
          "MBIC (outlet x topic)": mbic,
          "NewsWCL50 (outlet x target)": newswcl50,
          "NewsWCL50 without Democrats (a one-event target)": lambda: newswcl50(("Democrats",)),
          "AllSides adversarial (side x topic group)": lambda: allsides("outcome"),
          "AllSides directed (side x topic group)": lambda: allsides("outcome_dir")}
    R = {}
    for name, f in DS.items():
        c = certificate(f(), min_n=5, B=300, n_refine=50)
        R[name] = c
        loo = "  ".join(f"-{k}: p {v['p']:.3f}" for k, v in c["leave_one_source_out"].items())
        print(f"\n== {name}: {c['n_units']} units, {len(c['sources'])} sources x {len(c['topics'])} topics, {c['edges']} edges, "
              f"beta0 {c['beta0']}, beta1 {c['beta1']}")
        print(f"   spectral: Fiedler {c['fiedler']:.3g}  vs  q95 ||dL|| {c['dL_q95']:.3g}  -> margin {c['margin']:.2f}"
              f"  (bootstrap connected {c['frac_boot_connected']:.2f})")
        print(f"   obstruction: Q {c['Q']:.1f} on {c['beta1']} df, p {c['p']:.2g};  SNR {c['snr']:.2f};  median cos(h, h*) {c['median_cos']:.2f}")
        print(f"   refinement (topic cover split in halves, 50x): median p {c['refine_median_p']:.2g}, frac p<0.05 {c['refine_frac_sig']:.2f}")
        print(f"   leave one source out: {loo}")
    json.dump(R, open(os.path.join(here, "certificate_results.json"), "w"), indent=1, default=str)
