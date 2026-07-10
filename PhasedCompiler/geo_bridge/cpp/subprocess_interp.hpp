#pragma once
// subprocess_interp.hpp — executes the op stream with a CHECKPOINT BOUNDARY per
// phase. Each phase op spawns a fresh `phase_runner.py` process that loads the
// previous checkpoint, runs exactly one phase, and writes the next checkpoint.
//
// This proves the coordinate-driven claim: because phases communicate ONLY
// through the serialized model state (+ geometric invariants), running phase N
// in a separate process that loads phase N-1's .pt yields the same result as an
// in-process run. The handoff crosses a process boundary via the filesystem, so
// nothing GPU- or memory-resident leaks between phases -- which is exactly what
// lets a phase move to another machine/GPU.
//
// Control flow (loops, branches, asserts) runs in THIS process's VM, using the
// metrics each subprocess prints back. Only the heavy phase work is spawned out.
//
// No pybind11 here -- this interpreter is pure C++ + subprocess, so it needs no
// embedded interpreter at all. `python` is invoked as a child process.

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <memory>
#include <fstream>
#include <cctype>

namespace bridge {

class SubprocessInterpreter {
    std::vector<std::string> prog_;
    std::unordered_map<std::string, std::size_t> labels_;
    std::unordered_map<int, long> counters_;
    std::unordered_map<std::string, double> metrics_;
    std::unordered_map<std::string, std::vector<double>> history_;

    std::string python_;    // interpreter path
    std::string pydir_;     // dir containing phase_runner.py
    std::string ckptdir_;   // where phaseNN.pt live
    int seed_;
    std::string corpus_;    // "real" | "synthetic"
    int phase_idx_ = 0;     // increments each checkpoint-producing phase
    std::string results_;   // accumulated per-phase result lines (JSON array body)
    int op_seq_ = 0;        // sequence number across executed phase ops

    // phases that produce/consume a checkpoint (heavy, mutate the model)
    static const std::unordered_set<std::string>& checkpoint_ops() {
        static const std::unordered_set<std::string> s = {
            "SADDLE","MFPUMP","LANCZOS","BASIN","TAU_RETRY","SNAPPER",
            "TOPOGATE","ALIGN_LM","K0_SPLIT","JOINT_CE"
        };
        return s;
    }
    // sensor phases: read a metric, no model mutation, still a subprocess
    // (they load the current checkpoint read-only)
    static const std::unordered_set<std::string>& sensor_ops() {
        static const std::unordered_set<std::string> s = {
            "TAU","MEM","EVAL"
        };
        return s;
    }

    static std::vector<std::string> split(const std::string& s) {
        std::vector<std::string> out; std::istringstream is(s); std::string w;
        while (is >> w) out.push_back(w);
        return out;
    }
    static long kv(const std::string& tok) {
        auto eq = tok.find('='); return std::stol(tok.substr(eq + 1));
    }

    std::string ckpt_path(int idx) const {
        char buf[512];
        std::snprintf(buf, sizeof buf, "%s/phase%02d.pt", ckptdir_.c_str(), idx);
        return buf;
    }

    // Map an op line to a phase_runner --phase name and (optional) extra env.
    static std::string phase_name(const std::string& op) {
        if (op=="SADDLE") return "saddle";
        if (op=="MFPUMP") return "mfpump";
        if (op=="LANCZOS") return "lanczos";
        if (op=="BASIN") return "basin";
        if (op=="TAU_RETRY") return "tau_retry";
        if (op=="SNAPPER") return "snapper";
        if (op=="TOPOGATE") return "topogate";
        if (op=="ALIGN_LM") return "align_lm";
        if (op=="K0_SPLIT") return "k0_split";
        if (op=="JOINT_CE") return "joint_ce";
        if (op=="TAU") return "tau";
        if (op=="MEM") return "mem";
        if (op=="EVAL") return "eval";
        return "";
    }

    // Run a subprocess, capture stdout, parse the "METRICS {json}" line.
    // Returns the raw json string (without the METRICS prefix) or "" on failure.
    std::string run_subprocess(const std::string& cmd) {
        std::array<char, 4096> buf;
        std::string out;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) throw std::runtime_error("popen failed: " + cmd);
        while (fgets(buf.data(), (int)buf.size(), pipe)) out += buf.data();
        int rc = pclose(pipe);
        if (rc != 0) {
            std::fprintf(stderr, "phase subprocess failed (rc=%d):\n%s\n", rc, out.c_str());
            throw std::runtime_error("phase subprocess nonzero exit");
        }
        // find the METRICS line
        std::istringstream is(out); std::string line, metrics_json;
        while (std::getline(is, line)) {
            if (line.rfind("METRICS ", 0) == 0) metrics_json = line.substr(8);
            else if (!line.empty()) std::fprintf(stdout, "    %s\n", line.c_str()); // echo phase logs
        }
        return metrics_json;
    }

    // Minimal JSON parse for a flat {"k":num,...} object.
    static std::unordered_map<std::string,double> parse_metrics(const std::string& j) {
        std::unordered_map<std::string,double> m;
        std::size_t i = 0;
        while ((i = j.find('"', i)) != std::string::npos) {
            std::size_t k0 = i + 1, k1 = j.find('"', k0);
            if (k1 == std::string::npos) break;
            std::string key = j.substr(k0, k1 - k0);
            std::size_t colon = j.find(':', k1);
            if (colon == std::string::npos) break;
            std::size_t v0 = colon + 1;
            while (v0 < j.size() && (j[v0]==' ')) ++v0;
            std::size_t v1 = v0;
            while (v1 < j.size() && (std::isdigit((unsigned char)j[v1])||j[v1]=='.'||j[v1]=='-'||j[v1]=='+'||j[v1]=='e'||j[v1]=='E')) ++v1;
            try { m[key] = std::stod(j.substr(v0, v1 - v0)); } catch (...) {}
            i = v1;
        }
        return m;
    }

    void merge_metrics(const std::string& json) {
        auto m = parse_metrics(json);
        for (auto& [k,v] : m) {
            metrics_[k] = v;
            history_[k].push_back(v);
        }
    }

    // Spawn phase_runner for a heavy/sensor op. Threads the checkpoint chain:
    // in = current phase_idx, out = phase_idx+1 (for mutating ops).
    void spawn_phase(const std::string& op, const std::vector<std::string>& toks) {
        std::string phase = phase_name(op);
        bool mutates = checkpoint_ops().count(op) > 0;

        std::string in_ck  = ckpt_path(phase_idx_);
        std::string out_ck = mutates ? ckpt_path(phase_idx_ + 1) : "";

        std::string cmd = python_ + " " + pydir_ + "/phase_runner.py"
                        + " --phase " + phase
                        + " --in "  + in_ck
                        + " --seed " + std::to_string(seed_)
                        + " --corpus " + corpus_;
        if (!out_ck.empty()) cmd += " --out " + out_ck;

        // basin max via env
        std::string env_prefix;
        if (op == "BASIN") {
            long ms = toks.size() > 1 ? kv(toks[1]) : 150;
            env_prefix = "BASIN_MAX=" + std::to_string(ms) + " ";
        }
        std::string full = env_prefix + cmd;

        std::printf("  [%s] in=phase%02d%s\n", phase.c_str(), phase_idx_,
                    mutates ? (" -> phase" + std::to_string(phase_idx_+1)).c_str() : "");
        std::string mj = run_subprocess(full);
        merge_metrics(mj);

        // record a per-phase result line: {"seq":N,"phase":"...","in":.., "out":.., <metrics>}
        {
            std::string rec = "{\"seq\":" + std::to_string(op_seq_++)
                + ",\"phase\":\"" + phase + "\""
                + ",\"in\":\"phase" + (phase_idx_ < 10 ? "0" : "") + std::to_string(phase_idx_) + ".pt\"";
            if (mutates)
                rec += ",\"out\":\"phase" + std::string(phase_idx_+1 < 10 ? "0" : "")
                     + std::to_string(phase_idx_+1) + ".pt\"";
            // splice the metric json body (strip its braces) in
            std::string body = mj;
            if (body.size() >= 2 && body.front()=='{' && body.back()=='}')
                body = body.substr(1, body.size()-2);
            if (!body.empty()) rec += "," + body;
            rec += "}";
            if (!results_.empty()) results_ += ",\n";
            results_ += "  " + rec;
        }

        if (mutates) phase_idx_++;
    }

public:
    SubprocessInterpreter(std::vector<std::string> program,
                          std::string python, std::string pydir,
                          std::string ckptdir, int seed, std::string corpus)
        : prog_(std::move(program)), python_(std::move(python)),
          pydir_(std::move(pydir)), ckptdir_(std::move(ckptdir)),
          seed_(seed), corpus_(std::move(corpus)) {
        for (std::size_t pc = 0; pc < prog_.size(); ++pc) {
            auto t = split(prog_[pc]);
            if (!t.empty() && t[0] == "LABEL") labels_[t[1]] = pc;
        }
    }

    void run() {
        // init phase: build fresh model, spectral E0 + floor, write phase00.pt
        {
            std::string out0 = ckpt_path(0);
            std::string cmd = python_ + " " + pydir_ + "/phase_runner.py"
                            + " --phase init --out " + out0
                            + " --seed " + std::to_string(seed_)
                            + " --corpus " + corpus_;
            std::printf("[init] building phase00.pt (spectral E0 + floor)...\n");
            merge_metrics(run_subprocess(cmd));
            std::printf("[floor] corpus floor val = %.4f\n",
                        metrics_.count("floor") ? metrics_["floor"] : 0.0);
        }

        std::size_t pc = 0;
        while (pc < prog_.size()) {
            auto toks = split(prog_[pc]);
            if (toks.empty()) { ++pc; continue; }
            const std::string& op = toks[0];

            if (op == "HALT") break;
            else if (op == "LABEL" || op == "CONFIG") { ++pc; }
            else if (op == "LOG") {
                auto q1 = prog_[pc].find('"'), q2 = prog_[pc].rfind('"');
                std::printf("[log] %s\n",
                    (q1!=std::string::npos && q2>q1) ? prog_[pc].substr(q1+1,q2-q1-1).c_str() : "");
                ++pc;
            }
            else if (checkpoint_ops().count(op) || sensor_ops().count(op)) {
                spawn_phase(op, toks);
                ++pc;
            }
            else if (op == "SET_COUNTER") { counters_[std::stoi(toks[1])] = std::stol(toks[2]); ++pc; }
            else if (op == "DEC_COUNTER") { counters_[std::stoi(toks[1])]--; ++pc; }
            else if (op == "JZ_COUNTER") {
                if (counters_[std::stoi(toks[1])] <= 0) pc = labels_.at(toks[2]); else ++pc;
            }
            else if (op == "JMP") { pc = labels_.at(toks[1]); }
            else if (op == "RESET_RISING") { history_[toks[1]].clear(); ++pc; }
            else if (op == "BRANCH_LT") {
                auto it = metrics_.find(toks[1]);
                if (it!=metrics_.end() && it->second < std::stod(toks[2])) pc = labels_.at(toks[3]); else ++pc;
            }
            else if (op == "BRANCH_GE") {
                auto it = metrics_.find(toks[1]);
                if (it!=metrics_.end() && it->second >= std::stod(toks[2])) pc = labels_.at(toks[3]); else ++pc;
            }
            else if (op == "BRANCH_LT_M") {
                double a = metrics_.count(toks[1])?metrics_[toks[1]]:1e18;
                double b = metrics_.count(toks[2])?metrics_[toks[2]]:-1e18;
                if (a < b) pc = labels_.at(toks[3]); else ++pc;
            }
            else if (op == "BRANCH_GE_M") {
                double a = metrics_.count(toks[1])?metrics_[toks[1]]:-1e18;
                double b = metrics_.count(toks[2])?metrics_[toks[2]]:1e18;
                if (a >= b) pc = labels_.at(toks[3]); else ++pc;
            }
            else if (op == "BRANCH_RISING2") {
                auto& h = history_[toks[1]];
                bool r2 = h.size()>=3 && h[h.size()-1]>h[h.size()-2] && h[h.size()-2]>h[h.size()-3];
                if (r2) { std::printf("  [stop] %s rising twice\n", toks[1].c_str()); pc = labels_.at(toks[2]); }
                else ++pc;
            }
            else if (op == "ASSERT_LT") {
                double cur = metrics_.count(toks[1])?metrics_[toks[1]]:1e18;
                double v = std::stod(toks[2]);
                if (!(cur < v)) throw std::runtime_error("assertion failed: " + toks[1]);
                std::printf("  assert %s < %.4f (ok, %.4f)\n", toks[1].c_str(), v, cur);
                ++pc;
            }
            else { std::fprintf(stderr, "unknown op: %s\n", op.c_str()); ++pc; }
        }
        std::printf("\nfinal checkpoint: %s\n", ckpt_path(phase_idx_).c_str());

        // write results.json: full per-phase record + summary
        std::string path = ckptdir_ + "/results.json";
        std::ofstream out(path);
        out << "{\n";
        out << "  \"seed\": " << seed_ << ",\n";
        out << "  \"corpus\": \"" << corpus_ << "\",\n";
        out << "  \"floor\": " << (metrics_.count("floor") ? metrics_["floor"] : 0.0) << ",\n";
        out << "  \"final_val\": " << (metrics_.count("val") ? metrics_["val"] : 0.0) << ",\n";
        out << "  \"final_checkpoint\": \"" << ckpt_path(phase_idx_) << "\",\n";
        out << "  \"phases\": [\n" << results_ << "\n  ]\n";
        out << "}\n";
        out.close();
        std::printf("results written: %s\n", path.c_str());
        std::printf("  final_val=%.4f  floor=%.4f\n",
                    metrics_.count("val") ? metrics_["val"] : 0.0,
                    metrics_.count("floor") ? metrics_["floor"] : 0.0);
    }
};

} // namespace bridge
