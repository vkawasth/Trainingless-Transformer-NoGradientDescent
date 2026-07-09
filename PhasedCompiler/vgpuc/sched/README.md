# Training-schedule compiler + runtime

A second front-end for the project: a small DSL that describes a **training
schedule** and compiles to an op stream the C++ bridge executes against PyTorch.
This closes the loop — the `.sched` file is the *program*, `schedc` is the
*compiler*, the op stream is the *bytecode*, and `run_schedule` is the *VM* that
drives real training.

```
train.sched ──schedc──► train.ops ──run_schedule──► PyTorch (via TaskQueue+GIL)
 (source)      (compiler)  (op stream)   (VM)
```

## Language

```
config seed=<int> corpus=<synthetic|real>   # once, first
eval  n=<int>                                # -> $val
train steps=<int>                            # -> $loss
tau                                          # gluing defect -> $tau
mem                                          # allocated MiB -> $mem
log "<message>"
repeat <int> { ... }                         # counter loop
until $<metric> < <float> max <int> { ... }  # runtime-conditional loop, capped
assert $<metric> < <float>                   # halt schedule if violated
```

Metrics are `val`, `loss`, `tau`, `mem`. See `train.sched`.

## Compile

From `vgpuc/sched`:

```bash
clang++ -std=c++23 -O2 -I../include -Wall -Wextra schedc.cpp -o schedc
./schedc train.sched            # print op stream
./schedc train.sched train.ops  # write op stream
```

The compiler enforces: config first and once; positive counts/steps/iters; known
metric names; balanced braces — each as an `Expected<_,Diag>` with line:col.

## How loops compile

- `repeat N { body }` → a counter loop:
  `SET_COUNTER s N` / `LABEL top` / `JZ_COUNTER s end` / body / `DEC_COUNTER s`
  / `JMP top` / `LABEL end`.
- `until $m < v max N { body }` → the same counter cap PLUS a runtime early-exit
  `BRANCH_LT m v end` evaluated against the last observed metric. A data-dependent
  loop can't be unrolled, so it must compile to jumps + a runtime branch.

## Run

The op stream is executed by `run_schedule` (built by the bridge's CMake):

```bash
# from geo_bridge/
VENV_PYTHON=$(which python3) ./build/run_schedule ../vgpuc/sched/train.ops py
```

`run_schedule` reads the `CONFIG` line to build the `PyBridge` with the right
seed/corpus, then interprets the stream: TRAIN/EVAL/TAU/MEM ops are submitted to
the `TaskQueue` and their scalar results read back via `std::future`; control ops
(counters, jumps, branches, asserts) run in the VM. Same GIL discipline as
`geo_bridge`.
```
