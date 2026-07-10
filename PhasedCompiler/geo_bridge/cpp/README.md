# geo_bridge — C++ control plane driving the geometry compiler

A C++ front-end that drives the PyTorch geometry-compiler logic op-by-op through
a **worker-thread task queue**, reading results back via `std::future`, with an
**embedded CPython interpreter** (pybind11) and correct **GIL management**.

```
 C++ CONTROL PLANE                         PYTHON EXECUTION
 ┌───────────────────────────┐            ┌──────────────────────────┐
 │ main.cpp  (a schedule of  │            │ geo_compiler_surface.py  │
 │  ops: train/eval/tau/mem) │            │  GeometryCompiler:       │
 │        │ submit()         │  pybind11  │   .train_steps()         │
 │        ▼                  │  + GIL     │   .eval_val()            │
 │ TaskQueue (1 worker) ─────┼───────────►│   .gluing_defect()       │
 │        │ std::future      │            │   .mem_allocated_mib()   │
 │        ▼ .get()           │            │  (real LM + torch)       │
 │ log / branch on scalars   │            └──────────────────────────┘
 └───────────────────────────┘
```

## Files

- `py/geo_compiler_surface.py` — importable surface over the training logic.
  The original `compiler_geometri_patched_86_memfixed.py` is a run-on-import
  script; this refactors the reusable parts (`LM`, eval, step, tau) into a
  `GeometryCompiler` class whose methods are individual, future-friendly ops.
  Uses a synthetic in-memory Markov corpus by default so it imports instantly;
  pass `--real` (or `use_real_corpus(True)`) to load `/tmp/*.json`.
- `cpp/task_queue.hpp` — worker-thread queue, evolved from the single-threaded
  `TaskQueue` in `consteval_constexpr.cpp`. Thread-safe `submit()` returning
  `std::future<result>`, exception propagation, clean shutdown. Python-agnostic.
- `cpp/py_bridge.hpp` — embeds CPython, holds a `GeometryCompiler`, exposes ops.
  All GIL discipline lives here (see the big comment block).
- `cpp/main.cpp` — the driver / schedule.

## The GIL model (the important part)

CPython's Global Interpreter Lock means only one thread may touch Python objects
at a time. With a worker thread executing the ops, the discipline is:

1. The interpreter starts **on the main thread** (via `PyConfig` +
   `scoped_interpreter`), which then **holds** the GIL.
2. After building the compiler instance, the main thread **releases** the GIL
   for the bridge's lifetime (`py::gil_scoped_release` kept alive). Without this,
   the worker can never acquire the GIL → **deadlock**.
3. Every op body **acquires** the GIL (`py::gil_scoped_acquire`, RAII) before
   calling Python, releasing it on scope exit.
4. **Teardown order**: shut the queue down (worker stops calling Python) *before*
   the interpreter is destroyed. In `main.cpp` this is guaranteed by declaring
   `TaskQueue q` *after* `PyBridge py`, so `q` is destroyed first.

## Embedding a virtualenv correctly (the robust fix)

The embedded interpreter does **not** use whatever venv is "active" in your
shell — it computes its home from `libpython`'s location, which for a Homebrew
Python lands in the *base* install, not your venv. Pointing `PYTHONHOME` at the
venv root then breaks the stdlib (`Failed to import encodings module`), because a
venv contains only `site-packages`, not the full standard library.

The correct, shell-independent fix is to configure the interpreter in C++ via
`PyConfig`, setting `config.executable` to the **venv's own interpreter**
(e.g. `.../fact_env/bin/python3.14`). Given that path, CPython reads the adjacent
`pyvenv.cfg` and wires up BOTH the base stdlib and the venv site-packages, with
correct `sys.prefix` / `sys.base_prefix`. No `PYTHONHOME`, no `PYTHONPATH`, works
from any directory. This is implemented in `py_bridge.hpp`.

You provide the venv python path one of three ways (priority order):
  1. CLI: `./geo_bridge py --real /abs/path/to/venv/bin/python3.14`
  2. env:  `VENV_PYTHON=/abs/path/to/venv/bin/python3.14 ./geo_bridge py`
  3. the compiled-in default at the top of `main.cpp` (edit once for your box).

## Build

Requires: a C++23 compiler, and a Python venv with **torch**, **numpy**, and
**pybind11** installed.

```bash
# activate your venv (e.g. fact_env) so the tools below resolve to it
pip install pybind11 torch numpy

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$(which python3) \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++   # macOS Homebrew LLVM
cmake --build build

# run — pass the venv interpreter so embedding resolves torch/numpy:
VENV_PYTHON=$(which python3) ./build/geo_bridge py
# or bake the path into main.cpp's default and just:
./build/geo_bridge py
./build/geo_bridge py --real            # uses /tmp/{train,val,vocab}.json
```

Set `-DPython3_EXECUTABLE=$(which python3)` at configure time so the linked
`libpython` matches your venv's base Python (same major.minor).

## Expected output (synthetic corpus)

```
model params = <N>
baseline val = <~6.2 for vocab 512>

round  train_loss val        tau      mem_MiB
1      ...        ...         ...      ...
...
done; shutting down (queue first, then interpreter)
```

Validation loss should trend **down** across rounds because the synthetic corpus
is a learnable Markov chain (not random ids).

## Mapping back to vgpuc

The schedule in `main.cpp` is written directly in C++, but conceptually it *is*
a compiled command stream: `train` ↔ `write`, `eval`/`tau` ↔ `poll`,
checkpoint ↔ `kick`. A natural next step is to have `vgpuc` emit the schedule
and this bridge consume it — the DSL becomes the program, the queue the executor,
PyTorch the backend.
