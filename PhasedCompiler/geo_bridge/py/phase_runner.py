"""
phase_runner.py
===============
Runs ONE phase of the geometry compiler in an isolated process, with checkpoint
in/out. This is the relocatable unit of work: its entire interface is

    (input checkpoint .pt, phase name, args)  ->  (output checkpoint .pt, metrics)

Because the geometry compiler is COORDINATE-DRIVEN (phases communicate only
through model weights + geometric invariants, never through optimizer momentum,
data position, or GPU-resident state), a phase's result is fully captured by the
saved state_dict. Running phase N in a fresh process that loads phase N-1's
checkpoint therefore yields the SAME result as running both in one process --
which is exactly what lets you move between GPUs/machines between phases.

Usage (invoked by run_schedule, one process per phase):
    python phase_runner.py --phase saddle \
        --in  ckpt/phase02.pt --out ckpt/phase03.pt \
        --seed 99 --corpus real --meta ckpt/phase03.json

  --phase   one of: init eval saddle mfpump lanczos basin tau_retry
                    snapper topogate align_lm k0_split joint_ce phi tau mem floor
  --in      input checkpoint (omit for the 'init' phase, which builds fresh)
  --out     output checkpoint to write after the phase
  --meta    JSON file to write metrics into: {"val":..,"tau":..,"phi":..}

The process prints one JSON line to stdout: the metrics dict. run_schedule reads
that to update its VM metric state (val/tau/phi/floor) for branch decisions.
"""

import argparse, json, os, sys
import torch

# import the full phase implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_phases import GeometryPhases


def build_or_load(args):
    """Construct a GeometryPhases and restore model state from --in if given.

    The floor gradient / spectral init only need to be built ONCE (in the init
    phase). Later phases skip that expensive setup and just load the state_dict;
    they reconstruct g_floor lazily only if a phase needs it (align_lm).
    """
    need_floor = args.phase in ("init", "align_lm")
    gc = GeometryPhases(
        seed=args.seed,
        use_real_corpus=(args.corpus == "real"),
        build_floor=need_floor,
        floor_steps=(200 if args.corpus == "real" else 30),
    )
    if args.in_ckpt and os.path.exists(args.in_ckpt):
        blob = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
        gc.model.load_state_dict(blob["model"])
        gc._step = blob.get("step", 0)
        # carry forward the floor val so $floor stays defined across processes
        if "floor_val" in blob:
            gc._floor_val = blob["floor_val"]
        # carry geo_stopped flag (basin -> tau_retry handoff)
        if "geo_stopped" in blob:
            gc._geo_stopped = blob["geo_stopped"]
        # NOTE: we intentionally do NOT restore RNG state (see save comment).
        # Each phase draws fresh randomness; the coordinate-driven claim is that
        # the geometry of the loaded weights determines the outcome, not the
        # random path. RNG independence is a FEATURE of this test, not a bug.
    return gc


def run_phase(gc, phase):
    """Dispatch one phase; return a metrics dict. Only the metrics a phase
    naturally produces are populated; run_schedule merges them into VM state."""
    m = {}
    if phase == "init":
        m["val"] = gc.eval_val(8)
        m["floor"] = gc.floor_val()
    elif phase == "eval":
        m["val"] = gc.eval_val(8)
    elif phase == "saddle":
        m["val"] = gc.saddle()
    elif phase == "mfpump":
        m["val"] = gc.mfpump(0)
        m["phi"] = gc.phi_clean()
    elif phase == "lanczos":
        m["val"] = gc.lanczos()
    elif phase == "basin":
        m["val"] = gc.basin_settle(max_steps=args_max(gc))
        m["tau"] = gc.gluing_defect(6)
        m["phi"] = gc.phi_clean()
    elif phase == "tau_retry":
        m["val"] = gc.tau_retry()
        m["tau"] = gc.gluing_defect(6)
    elif phase == "snapper":
        m["val"] = gc.snapper_jump()
    elif phase == "topogate":
        m["val"] = gc.topogate()
        m["phi"] = gc.phi_clean()
    elif phase == "align_lm":
        m["val"] = gc.align_lm()
        m["tau"] = gc.gluing_defect(6)
    elif phase == "k0_split":
        m["val"] = gc.k0_split()
    elif phase == "joint_ce":
        m["val"] = gc.joint_ce()
    elif phase == "tau":
        m["tau"] = gc.gluing_defect(6)
    elif phase == "phi":
        m["phi"] = gc.phi_clean()
    elif phase == "mem":
        m["mem"] = gc.mem_allocated_mib()
    elif phase == "floor":
        m["floor"] = gc.floor_val()
    else:
        raise SystemExit(f"unknown phase '{phase}'")
    return m


# basin max-steps is passed via env to keep the CLI simple
def args_max(gc):
    return int(os.environ.get("BASIN_MAX", "150"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--in",  dest="in_ckpt", default="")
    ap.add_argument("--out", dest="out_ckpt", default="")
    ap.add_argument("--meta", default="")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic"])
    args = ap.parse_args()

    gc = build_or_load(args)
    metrics = run_phase(gc, args.phase)

    # Measure the boundary-interface coordinates (Phi, sigma, tau, E) on the
    # phase's OUTPUT state. These are the geometric "type" of the checkpoint --
    # recorded into both the metrics (-> results.json) and the checkpoint itself,
    # so the invariants travel with the coordinates and can be asserted by the
    # next phase or compared across execution modes.
    geom = gc.geometry_probe()
    metrics["geom"] = geom

    # persist ONLY the coordinate state (weights + geometric scalars). We
    # deliberately do NOT checkpoint the RNG stream. Copying the RNG would make
    # a phase's output depend on the full dynamical state (like gradient
    # descent), which would FALSIFY the coordinate-driven claim: the whole point
    # is that a phase's result is determined by the geometry of its input, not
    # by the random path taken. Two runs from the same checkpoint SHOULD explore
    # different random directions and still land in the same geometric basin --
    # that's the property worth testing (endpoint geometry), not bit-identity.
    if args.out_ckpt:
        os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
        torch.save({
            "model": gc.model.state_dict(),
            "step": gc._step,
            "floor_val": float(getattr(gc, "_floor_val", 0.0)),
            "geo_stopped": bool(getattr(gc, "_geo_stopped", False)),
            "geom": geom,   # (phi, sigma, tau, E) boundary coordinates
        }, args.out_ckpt)

    if args.meta:
        with open(args.meta, "w") as f:
            json.dump(metrics, f)

    # machine-readable line for run_schedule / run_distributed. The C++ metric
    # parser is flat, so we emit the geometry coordinates as flat keys
    # (geom_phi, geom_sigma, geom_tau) alongside the phase's own metrics. E is
    # omitted while its definition is pending (kept in the JSON checkpoint/meta).
    flat = {k: v for k, v in metrics.items() if k != "geom"}
    flat["geom_phi"]   = geom["phi"]
    flat["geom_sigma"] = geom["sigma"]
    flat["geom_tau"]   = geom["tau"]
    print("METRICS " + json.dumps(flat))


if __name__ == "__main__":
    main()
