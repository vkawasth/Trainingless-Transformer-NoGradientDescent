#pragma once
// op_interp.hpp — executes a compiled schedule op-stream against the PyBridge,
// submitting each op to the TaskQueue and reading results back via futures.
//
// This is the runtime half of the "real compiler": schedc.cpp compiles
// train.sched -> op stream; this interprets the op stream, driving PyTorch.
//
// The op stream is line-oriented text (see sched_ir.hpp). We keep a tiny VM:
//   - counters[]           : integer loop counters
//   - labels{name->pc}     : resolved on a first pass
//   - metrics{name->value} : last observed val/loss/tau/mem
// Control ops (JMP/JZ_COUNTER/BRANCH_LT/SET/DEC) manipulate pc and counters.
// Work ops (TRAIN/EVAL/TAU/MEM) go through the queue to the bridge.

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <cstdio>

#include "task_queue.hpp"
#include "py_bridge.hpp"

namespace bridge {

class OpInterpreter {
    PyBridge&  py_;
    TaskQueue& q_;
    std::vector<std::string> prog_;
    std::unordered_map<std::string, std::size_t> labels_;
    std::unordered_map<int, long> counters_;
    std::unordered_map<std::string, double> metrics_;
    // history for rising-detection (BRANCH_RISING2): last two values per metric
    std::unordered_map<std::string, std::vector<double>> history_;

    static std::vector<std::string> split(const std::string& s) {
        std::vector<std::string> out; std::istringstream is(s); std::string w;
        while (is >> w) out.push_back(w);
        return out;
    }

    // parse "key=value" -> value (int)
    static long kv(const std::string& tok) {
        auto eq = tok.find('=');
        return std::stol(tok.substr(eq + 1));
    }

public:
    OpInterpreter(PyBridge& py, TaskQueue& q, std::vector<std::string> program)
        : py_(py), q_(q), prog_(std::move(program)) {
        // pass 1: resolve labels to program counters
        for (std::size_t pc = 0; pc < prog_.size(); ++pc) {
            auto toks = split(prog_[pc]);
            if (!toks.empty() && toks[0] == "LABEL")
                labels_[toks[1]] = pc;
        }
    }

    void run() {
        std::size_t pc = 0;
        while (pc < prog_.size()) {
            auto toks = split(prog_[pc]);
            if (toks.empty()) { ++pc; continue; }
            const std::string& op = toks[0];

            if (op == "HALT") break;
            else if (op == "LABEL") { ++pc; }
            else if (op == "CONFIG") {
                // config is applied at bridge construction; log for visibility
                std::printf("[config] %s %s\n", toks[1].c_str(), toks[2].c_str());
                ++pc;
            }
            else if (op == "LOG") {
                // reconstruct quoted message
                auto q1 = prog_[pc].find('"'), q2 = prog_[pc].rfind('"');
                std::printf("[log] %s\n",
                            (q1 != std::string::npos && q2 > q1)
                                ? prog_[pc].substr(q1 + 1, q2 - q1 - 1).c_str() : "");
                ++pc;
            }
            else if (op == "TRAIN") {
                int steps = (int)kv(toks[1]);
                double loss = q_.submit([&]{ return py_.train_steps(steps); }).get();
                metrics_["loss"] = loss;
                std::printf("  train(%d) -> loss=%.4f\n", steps, loss);
                ++pc;
            }
            else if (op == "EVAL") {
                int n = (int)kv(toks[1]);
                double val = q_.submit([&]{ return py_.eval_val(n); }).get();
                metrics_["val"] = val;
                std::printf("  eval(%d)  -> val=%.4f\n", n, val);
                ++pc;
            }
            else if (op == "TAU") {
                double t = q_.submit([&]{ return py_.gluing_defect(6); }).get();
                metrics_["tau"] = t;
                history_["tau"].push_back(t);
                std::printf("  tau      -> %.4f\n", t);
                ++pc;
            }
            else if (op == "SADDLE") {
                double v = q_.submit([&]{ return py_.saddle(); }).get();
                metrics_["val"] = v;
                std::printf("  saddle   -> val=%.4f\n", v);
                ++pc;
            }
            else if (op == "MFPUMP") {
                int seed = toks.size() > 1 ? (int)kv(toks[1]) : 0;
                double v = q_.submit([&]{ return py_.mfpump(seed); }).get();
                metrics_["val"] = v;
                // also refresh phi so guard $phi == 5 can fire
                double phi = q_.submit([&]{ return py_.phi_clean(); }).get();
                metrics_["phi"] = phi;
                std::printf("  mfpump   -> val=%.4f  phi=%.0f\n", v, phi);
                ++pc;
            }
            else if (op == "LANCZOS") {
                double v = q_.submit([&]{ return py_.lanczos(); }).get();
                metrics_["val"] = v;
                std::printf("  lanczos  -> val=%.4f\n", v);
                ++pc;
            }
            else if (op == "BASIN") {
                int ms = toks.size() > 1 ? (int)kv(toks[1]) : 150;
                double v = q_.submit([&]{ return py_.basin_settle(ms); }).get();
                double t = q_.submit([&]{ return py_.gluing_defect(6); }).get();
                double phi = q_.submit([&]{ return py_.phi_clean(); }).get();
                metrics_["val"] = v; metrics_["tau"] = t; metrics_["phi"] = phi;
                std::printf("  basin    -> val=%.4f  tau=%.2f  phi=%.0f\n", v, t, phi);
                ++pc;
            }
            else if (op == "TAU_RETRY") {
                double v = q_.submit([&]{ return py_.tau_retry(); }).get();
                double t = q_.submit([&]{ return py_.gluing_defect(6); }).get();
                metrics_["val"] = v; metrics_["tau"] = t;
                std::printf("  tau_retry-> val=%.4f  tau=%.2f\n", v, t);
                ++pc;
            }
            else if (op == "SNAPPER") {
                double v = q_.submit([&]{ return py_.snapper_jump(); }).get();
                metrics_["val"] = v;
                std::printf("  snapper  -> val=%.4f\n", v);
                ++pc;
            }
            else if (op == "TOPOGATE") {
                double v = q_.submit([&]{ return py_.topogate(); }).get();
                double phi = q_.submit([&]{ return py_.phi_clean(); }).get();
                metrics_["val"] = v; metrics_["phi"] = phi;
                std::printf("  topogate -> val=%.4f  phi=%.0f\n", v, phi);
                ++pc;
            }
            else if (op == "ALIGN_LM") {
                double v = q_.submit([&]{ return py_.align_lm(); }).get();
                double t = q_.submit([&]{ return py_.gluing_defect(6); }).get();
                metrics_["val"] = v; metrics_["tau"] = t;
                std::printf("  align_lm -> val=%.4f  tau=%.2f\n", v, t);
                ++pc;
            }
            else if (op == "K0_SPLIT") {
                double v = q_.submit([&]{ return py_.k0_split(); }).get();
                metrics_["val"] = v;
                std::printf("  k0_split -> val=%.4f\n", v);
                ++pc;
            }
            else if (op == "JOINT_CE") {
                double v = q_.submit([&]{ return py_.joint_ce(); }).get();
                metrics_["val"] = v;
                std::printf("  joint_ce -> val=%.4f\n", v);
                ++pc;
            }
            else if (op == "MEM") {
                double m = q_.submit([&]{ return py_.mem_allocated_mib(); }).get();
                metrics_["mem"] = m;
                std::printf("  mem      -> %.2f MiB\n", m);
                ++pc;
            }
            else if (op == "SET_COUNTER") {
                counters_[std::stoi(toks[1])] = std::stol(toks[2]);
                ++pc;
            }
            else if (op == "DEC_COUNTER") {
                counters_[std::stoi(toks[1])]--;
                ++pc;
            }
            else if (op == "JZ_COUNTER") {
                int slot = std::stoi(toks[1]);
                if (counters_[slot] <= 0) pc = labels_.at(toks[2]);
                else ++pc;
            }
            else if (op == "JMP") {
                pc = labels_.at(toks[1]);
            }
            else if (op == "BRANCH_LT") {
                // if last <metric> < value, jump to target
                const std::string& m = toks[1];
                double v = std::stod(toks[2]);
                auto it = metrics_.find(m);
                if (it != metrics_.end() && it->second < v) pc = labels_.at(toks[3]);
                else ++pc;
            }
            else if (op == "BRANCH_GE") {
                // if last <metric> >= value, jump to target (guard: phi == 5)
                const std::string& m = toks[1];
                double v = std::stod(toks[2]);
                auto it = metrics_.find(m);
                if (it != metrics_.end() && it->second >= v) pc = labels_.at(toks[3]);
                else ++pc;
            }
            else if (op == "RESET_RISING") {
                history_[toks[1]].clear();
                ++pc;
            }
            else if (op == "BRANCH_RISING2") {
                // jump if <metric> rose for two consecutive iterations:
                //   h[-1] > h[-2] > h[-3]   (matches the original's tau check)
                const std::string& m = toks[1];
                auto& h = history_[m];
                bool rising2 = h.size() >= 3 &&
                               h[h.size()-1] > h[h.size()-2] &&
                               h[h.size()-2] > h[h.size()-3];
                if (rising2) {
                    std::printf("  [stop] %s rising twice (%.2f->%.2f->%.2f)\n",
                                m.c_str(), h[h.size()-3], h[h.size()-2], h[h.size()-1]);
                    pc = labels_.at(toks[2]);
                } else ++pc;
            }
            else if (op == "ASSERT_LT") {
                const std::string& m = toks[1];
                double v = std::stod(toks[2]);
                auto it = metrics_.find(m);
                double cur = (it != metrics_.end()) ? it->second : 1e18;
                if (!(cur < v)) {
                    std::printf("  ASSERT FAILED: %s=%.4f not < %.4f\n", m.c_str(), cur, v);
                    throw std::runtime_error("schedule assertion failed");
                }
                std::printf("  assert %s < %.4f  (ok, %s=%.4f)\n",
                            m.c_str(), v, m.c_str(), cur);
                ++pc;
            }
            else {
                std::fprintf(stderr, "unknown op: %s\n", op.c_str());
                ++pc;
            }
        }
    }

    // Extract config from the op stream (first CONFIG line) so the driver can
    // build the PyBridge with the right seed/corpus before interpreting.
    struct ConfigInfo { int seed = 99; bool real = false; bool found = false; };
    static ConfigInfo peek_config(const std::vector<std::string>& prog) {
        ConfigInfo ci;
        for (const auto& l : prog) {
            std::istringstream is(l); std::string w; is >> w;
            if (w == "CONFIG") {
                std::string a, b; is >> a >> b;
                if (a.rfind("seed=",0)==0)   ci.seed = std::stoi(a.substr(5));
                if (b.rfind("corpus=",0)==0) ci.real = (b.substr(7)=="real");
                ci.found = true;
                break;
            }
        }
        return ci;
    }
};

} // namespace bridge
