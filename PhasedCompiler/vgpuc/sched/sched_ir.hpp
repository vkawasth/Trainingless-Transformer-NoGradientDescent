#pragma once
// sched_ir.hpp — IR for the training-schedule language.
//
// Source commands compile to a flat op stream (like a tiny bytecode). Control
// flow (repeat / until) becomes explicit jumps so that data-dependent loops
// (until $val < x) work at runtime -- we cannot unroll a loop whose exit
// depends on a metric produced during execution.
//
// The emitted stream is line-oriented text, one op per line, so it is trivially
// consumable by the C++ bridge (and human-readable for debugging):
//
//   CONFIG seed=99 corpus=synthetic
//   EVAL n=4
//   TRAIN steps=20
//   TAU
//   MEM
//   LOG "message"
//   JZ_COUNTER <slot> <target>      ; if counter[slot]-- == 0 jump to target
//   JMP <target>
//   BRANCH_GE <metric> <value> <target>  ; if last <metric> >= value, jump
//   ASSERT_LT <metric> <value>
//   HALT

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include "vgpuc/diag.hpp"

namespace sched {

using vgpuc::SourcePos;
using vgpuc::Diag;

// ---- source-level AST ----
enum class Corpus { Synthetic, Real };

struct Config { int seed; Corpus corpus; SourcePos pos; };
struct Eval   { int n; SourcePos pos; };
struct Train  { int steps; SourcePos pos; };
struct Tau    { SourcePos pos; };
struct Mem    { SourcePos pos; };
struct Saddle  { SourcePos pos; };
struct MfPump  { int seed; SourcePos pos; };
struct Lanczos { SourcePos pos; };
struct Basin    { int max_steps; SourcePos pos; };
struct TauRetry { SourcePos pos; };
struct Snapper  { SourcePos pos; };
struct TopoGate { SourcePos pos; };
struct AlignLM  { SourcePos pos; };
struct K0Split  { SourcePos pos; };
struct JointCE  { SourcePos pos; };
struct Log    { std::string message; SourcePos pos; };
struct Assert { std::string metric; double value; SourcePos pos; };

struct Block;  // fwd

struct Repeat { int count; std::vector<struct Stmt> body; SourcePos pos; };
struct Until  { std::string metric; double value; int max_iters;
                std::vector<struct Stmt> body; SourcePos pos; };
// Loop body up to max_iters, but STOP early when <metric> rises for two
// consecutive iterations (the original MF-pump "orbit shattering" sensor),
// OR when <guard_metric> reaches guard_value (e.g. phi == 5). guard optional.
struct PumpUntil { std::string rising_metric;    // e.g. "tau"
                   std::string guard_metric;     // e.g. "phi" ("" if none)
                   double      guard_value;      // e.g. 5
                   int         max_iters;
                   std::vector<struct Stmt> body; SourcePos pos; };

// if/elif/else on metric comparisons. Each arm has a condition (metric, "<" vs
// ">=", value) and a body; the else arm has an empty metric string.
struct CondArm { std::string metric; bool is_lt; double value;
                 std::vector<struct Stmt> body; };
struct If { std::vector<CondArm> arms; SourcePos pos; };

using StmtData = std::variant<Config, Eval, Train, Tau, Mem, Saddle, MfPump,
                              Lanczos, Basin, TauRetry, Snapper, TopoGate,
                              AlignLM, K0Split, JointCE, Log, Assert,
                              Repeat, Until, PumpUntil, If>;
struct Stmt { StmtData data; };

struct ScheduleAst { std::vector<Stmt> stmts; };

// ---- emitted op stream ----
// We keep it as text lines; the emitter builds these strings directly.

} // namespace sched
