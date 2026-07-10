// run_schedule.cpp — the "real compiler" runtime.
//   schedc  train.sched  train.ops     # compile (from vgpuc/sched)
//   run_schedule  train.ops  py        # execute against PyTorch
//
// Reads a compiled op stream, extracts config, builds the PyBridge with the
// right seed/corpus, then interprets the stream (driving PyTorch via the queue).

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "task_queue.hpp"
#include "py_bridge.hpp"
#include "op_interp.hpp"

static std::vector<std::string> read_lines(const char* path) {
    std::ifstream f(path);
    std::vector<std::string> out; std::string line;
    while (std::getline(f, line)) out.push_back(line);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: run_schedule <ops-file> [pydir] [module]\n"); return 2; }
    const std::string opsfile = argv[1];
    const std::string pydir   = (argc > 2) ? argv[2] : "../py";
    // module: which Python surface to import. Default is the full geometry
    // surface (saddle/mfpump/lanczos + basics). Pass "geo_compiler_surface"
    // for the lighter one. Can also be set via SCHED_MODULE env var.
    std::string module = (argc > 3) ? argv[3] : "";
    if (module.empty()) {
        if (const char* m = std::getenv("SCHED_MODULE"); m && *m) module = m;
        else module = "geo_phases";
    }

    std::vector<std::string> prog = read_lines(opsfile.c_str());

    // Resolve venv python (same policy as main.cpp).
    std::string venv_python;
    if (const char* e = std::getenv("VENV_PYTHON"); e && *e) venv_python = e;
    else venv_python =
        "/Users/vaw1/Downloads/OGB/connectome/"
        "phaseTransition_phaseTransition_complex/FukayaAUComplex/"
        "longChainHllucinationp-adic/j-holomorphic-Fukaya/fact_env/bin/python3.14";

    // Config comes from the compiled program itself.
    auto ci = bridge::OpInterpreter::peek_config(prog);
    // Allow a seed override via SCHED_SEED so a harness can vary runs.
    int seed = ci.seed;
    if (const char* s = std::getenv("SCHED_SEED"); s && *s) seed = std::atoi(s);
    std::printf("schedule config: seed=%d corpus=%s\n",
                seed, ci.real ? "real" : "synthetic");

    // Declare py before q so q (worker) is destroyed first (see teardown notes).
    bridge::PyBridge py(module, pydir, venv_python, ci.real, seed);
    bridge::TaskQueue q;

    bridge::OpInterpreter interp(py, q, std::move(prog));
    try {
        interp.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nschedule halted: %s\n", e.what());
        return 1;
    }
    std::printf("\nschedule complete\n");
    return 0;
}
