# Distributed (checkpoint-boundary) execution

The geometry compiler is **coordinate-driven**: phases communicate only through
model weights + geometric invariants (Φ, τ, alignment), never through optimizer
momentum, data position, or GPU-resident state. That means a phase's entire
output is captured by its serialized `state_dict` — so phase N can run in a
different process (or GPU, or machine) that loads phase N-1's checkpoint, and the
result is identical to an in-process run.

This is what `run_distributed` demonstrates: **one subprocess per phase**, with a
`.pt` checkpoint as the only handoff between them.

```
init ─► phase00.pt ─► SADDLE ─► phase01.pt ─► MFPUMP ─► phase02.pt ─► ... ─► final
        (proc 1)      (proc 2)                (proc 3)
control flow (loops/branches/asserts) runs in the C++ VM;
coordinate state lives entirely in phaseNN.pt files.
```

## Two execution modes

| | `run_schedule` | `run_distributed` |
|---|---|---|
| interpreter | one embedded CPython, all phases in-process | one subprocess per phase |
| phase handoff | in-memory Python object | `phaseNN.pt` on disk |
| needs pybind11 | yes | no (pure C++ + subprocess) |
| proves | GIL/queue/embedding | coordinate-driven relocatability |
| speed | faster (no reload) | slower (reload per phase) but relocatable |

Both run the SAME compiled `geo.ops`. Same control flow, same branches.

## Run it

```bash
# build (run_distributed is a separate CMake target, no pybind needed)
cmake --build build

mkdir -p ckpt
VENV_PYTHON=$(which python3) \
  ./build/run_distributed ../vgpuc/sched/geo.ops py ckpt 99 real
```

After it runs, `ckpt/` holds one checkpoint per phase boundary
(`phase00.pt` ... `phaseNN.pt`). You can inspect any of them independently, or —
the whole point — copy `phase03.pt` to another machine and resume from Phase 4
there:

```bash
# on machine B, continue from a mid-pipeline checkpoint:
python py/phase_runner.py --phase basin \
    --in  ckpt/phase04.pt --out ckpt/phase05.pt \
    --seed 99 --corpus real
```

## Results

After a distributed run, `<ckptdir>/results.json` holds the full trace:
```json
{
  "seed": 99, "corpus": "real",
  "floor": 0.062, "final_val": 0.046,
  "final_checkpoint": "ckpt/phase07.pt",
  "phases": [
    {"seq":1,"phase":"saddle","in":"phase00.pt","out":"phase01.pt","val":4.49},
    ...
  ]
}
```
Plus the checkpoints themselves (`phaseNN.pt`), one per phase boundary.

## Verifying the coordinate-driven claim

`compare_modes.sh` is the thesis test: it runs the SAME seed both ways and checks
the final vals agree.

```bash
./compare_modes.sh 99 real
```

If the compiler is truly coordinate-driven, the in-process result (all phases
sharing memory) and the distributed result (each phase a separate process, state
crossing only via `.pt` files) must match within stochastic tolerance. A match
proves the process boundary loses nothing — phases can move between GPUs.

## phase_runner.py

The relocatable unit. Interface:
```
python phase_runner.py --phase <name> --in <ckpt> --out <ckpt> \
    --seed <n> --corpus <real|synthetic>
```
Loads `--in`, runs exactly one phase, saves `--out`, prints `METRICS {json}`.
`run_distributed` reads that JSON to update the VM's metric state (val/tau/phi/
floor) for branch decisions.

Because the checkpoint carries `floor_val` and `geo_stopped` alongside the model
weights, cross-process handoffs preserve the small amount of scalar state the
schedule branches on (e.g. basin's geo-stop flag feeding tau_retry).
