// run_distributed.cpp — run the geometry compiler with a CHECKPOINT BOUNDARY
// per phase, each phase in a fresh subprocess. Pure C++ + subprocess (no
// embedded interpreter needed).
//
//   run_distributed <ops-file> <pydir> <ckptdir> [seed] [corpus]
//
// Proves the coordinate-driven handoff: phases exchange only phaseNN.pt files,
// so each could run on a different GPU/machine. Here they run as separate local
// processes sharing a checkpoint directory.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "subprocess_interp.hpp"

static std::vector<std::string> read_lines(const char* p) {
    std::ifstream f(p); std::vector<std::string> v; std::string l;
    while (std::getline(f, l)) v.push_back(l);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: run_distributed <ops-file> <pydir> <ckptdir> [seed] [corpus]\n");
        return 2;
    }
    std::string ops = argv[1], pydir = argv[2], ckptdir = argv[3];
    int seed = (argc > 4) ? std::atoi(argv[4]) : 99;
    std::string corpus = (argc > 5) ? argv[5] : "real";

    // python interpreter: prefer VENV_PYTHON, else `python3` on PATH.
    std::string python = "python3";
    if (const char* e = std::getenv("VENV_PYTHON"); e && *e) python = e;

    auto prog = read_lines(ops.c_str());
    if (prog.empty()) { std::fprintf(stderr, "empty ops file\n"); return 1; }

    std::printf("distributed run: seed=%d corpus=%s ckptdir=%s\n",
                seed, corpus.c_str(), ckptdir.c_str());
    std::printf("python=%s\n\n", python.c_str());

    bridge::SubprocessInterpreter interp(std::move(prog), python, pydir, ckptdir,
                                         seed, corpus);
    try {
        interp.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nrun halted: %s\n", e.what());
        return 1;
    }
    std::printf("distributed run complete\n");
    return 0;
}
