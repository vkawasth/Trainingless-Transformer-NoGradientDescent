// main.cpp
// C++ control plane driving the geometry compiler through a worker-thread
// TaskQueue, reading results back via std::future.
//
// This mirrors the vgpuc idea: a schedule of ops (here written directly in C++,
// but conceptually the compiled command stream) is submitted to an executor;
// the executor runs each op against the PyTorch backend and returns scalars the
// control plane logs and branches on.
//
// Build: see CMakeLists.txt (needs pybind11 + a Python with torch/numpy).

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <future>
#include <string>

#include "task_queue.hpp"
#include "py_bridge.hpp"

int main(int argc, char** argv) {
    const std::string module = "geo_compiler_surface";
    const std::string pydir  = (argc > 1) ? argv[1] : "../py";
    const bool real_corpus   = (argc > 2) && std::string(argv[2]) == "--real";

    // Resolve the venv interpreter path, in priority order:
    //   1) 3rd CLI arg   ./geo_bridge py --real /path/to/venv/bin/python3.14
    //      (or 2nd arg if you skip --real; see below)
    //   2) VENV_PYTHON environment variable
    //   3) a compiled-in default (edit for your machine)
    // Pointing the embedded interpreter at the venv's own python makes CPython
    // read pyvenv.cfg and resolve base-stdlib + venv site-packages correctly,
    // with no PYTHONHOME/PYTHONPATH needed.
    std::string venv_python;
    if (const char* env = std::getenv("VENV_PYTHON"); env && *env) {
        venv_python = env;
    } else {
        // EDIT THIS DEFAULT to your venv's interpreter:
        venv_python =
            "/Users/vaw1/Downloads/OGB/connectome/"
            "phaseTransition_phaseTransition_complex/FukayaAUComplex/"
            "longChainHllucinationp-adic/j-holomorphic-Fukaya/fact_env/bin/python3.14";
    }

    // 1) Start the interpreter + build the compiler instance (main thread).
    //    Constructing PyBridge holds then releases the GIL (see py_bridge.hpp).
    bridge::PyBridge py(module, pydir, venv_python, real_corpus);

    // 2) Start the executor.
    bridge::TaskQueue q;

    // Helper: submit an op that calls into py on the worker thread. Note the
    // capture of &py by reference -- the bridge outlives the queue (see teardown
    // ordering at the end).
    auto eval  = [&](int n)  { return q.submit([&py, n] { return py.eval_val(n); }); };
    auto train = [&](int k)  { return q.submit([&py, k] { return py.train_steps(k); }); };
    auto tau   = [&]()       { return q.submit([&py]    { return py.gluing_defect(6); }); };
    auto mem   = [&]()       { return q.submit([&py]    { return py.mem_allocated_mib(); }); };

    // 3) One-time info (blocks on the future immediately, which is fine).
    long np = q.submit([&py]{ return py.num_params(); }).get();
    std::printf("model params = %ld\n", np);

    // 4) A schedule: baseline eval, then N rounds of {train, eval, tau, mem}.
    //    We submit the whole round, THEN read futures -- so submission isn't
    //    serialized by our own .get() calls (though the worker runs them in
    //    order anyway; this is about not blocking the submitting thread).
    double v0 = eval(4).get();
    std::printf("baseline val = %.4f\n", v0);

    const int rounds = 5;
    std::printf("\n%-6s %-10s %-10s %-8s %-10s\n", "round", "train_loss", "val", "tau", "mem_MiB");
    for (int r = 1; r <= rounds; ++r) {
        auto f_train = train(20);
        auto f_val   = eval(4);
        auto f_tau   = tau();
        auto f_mem   = mem();

        // Reading futures blocks until each op completes on the worker.
        double tl = f_train.get();
        double v  = f_val.get();
        double t  = f_tau.get();
        double m  = f_mem.get();
        std::printf("%-6d %-10.4f %-10.4f %-8.3f %-10.2f\n", r, tl, v, t, m);
    }

    // 5) TEARDOWN ORDER MATTERS:
    //    - q goes out of scope first (its dtor shuts down the worker thread,
    //      so no task will touch Python after this point),
    //    - THEN py (PyBridge) is destroyed, re-acquiring the GIL on the main
    //      thread and tearing the interpreter down on its creating thread.
    //    Because `q` is declared after `py`, C++ destroys q BEFORE py. Good.
    std::printf("\ndone; shutting down (queue first, then interpreter)\n");
    return 0;
}
