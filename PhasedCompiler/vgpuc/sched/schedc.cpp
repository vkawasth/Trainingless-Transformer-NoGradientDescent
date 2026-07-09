// schedc.cpp — compiler for the training-schedule language.
//
// Pipeline mirrors vgpuc: read source -> parse (Expected<Ast,Diag>) ->
// check -> emit op stream. Loops compile to explicit jumps so runtime-
// conditional loops (until $metric) work.
//
// Build (standalone; reuses only vgpuc's diag.hpp + expected.hpp headers):
//   clang++ -std=c++23 -O2 -I../include -Wall -Wextra schedc.cpp -o schedc
// Usage:
//   ./schedc train.sched            # prints op stream to stdout
//   ./schedc train.sched out.ops    # writes op stream to file

#include <cctype>
#include <cstdio>
#include <charconv>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "vgpuc/expected.hpp"
#include "sched_ir.hpp"

using vgpuc::Expected;
using vgpuc::Unexpected;
using vgpuc::Diag;
using vgpuc::SourcePos;
using vgpuc::diag;
using namespace sched;

// ------------------------------- lexer --------------------------------------
namespace {

struct Tok {
    enum Kind { Ident, Int, Float, Str, LBrace, RBrace, Eq, Dollar, Lt, End } kind;
    std::string text;
    double fval = 0;
    long   ival = 0;
    SourcePos pos;
};

struct Lexer {
    std::string_view s;
    std::size_t i = 0;
    std::uint32_t line = 1, col = 1;

    char peek(std::size_t k = 0) const { return i + k < s.size() ? s[i + k] : '\0'; }
    char get() { char c = s[i++]; if (c=='\n'){line++;col=1;} else col++; return c; }
    bool eof() const { return i >= s.size(); }
    SourcePos here() const { return {line, col}; }

    Expected<std::vector<Tok>, Diag> run() {
        std::vector<Tok> out;
        while (!eof()) {
            char c = peek();
            if (c==' '||c=='\t'||c=='\r'||c=='\n') { get(); continue; }
            if (c=='#') { while(!eof() && peek()!='\n') get(); continue; }
            SourcePos p = here();
            if (c=='{') { get(); out.push_back({Tok::LBrace,"{",0,0,p}); continue; }
            if (c=='}') { get(); out.push_back({Tok::RBrace,"}",0,0,p}); continue; }
            if (c=='=') { get(); out.push_back({Tok::Eq,"=",0,0,p}); continue; }
            if (c=='$') { get(); out.push_back({Tok::Dollar,"$",0,0,p}); continue; }
            if (c=='<') { get(); out.push_back({Tok::Lt,"<",0,0,p}); continue; }
            if (c=='"') {
                get(); std::string str;
                while(!eof() && peek()!='"') str += get();
                if (eof()) return Unexpected<Diag>{diag("unterminated string", p)};
                get();
                out.push_back({Tok::Str, str, 0, 0, p}); continue;
            }
            if (std::isdigit((unsigned char)c) || (c=='-' && std::isdigit((unsigned char)peek(1)))) {
                std::size_t b = i; bool isf=false;
                if (peek()=='-') get();
                while(!eof() && (std::isdigit((unsigned char)peek())||peek()=='.')) {
                    if (peek()=='.') isf=true;
                    get();
                }
                std::string num(s.substr(b, i-b));
                Tok t; t.pos=p;
                if (isf) { t.kind=Tok::Float; t.fval=std::stod(num); }
                else     { t.kind=Tok::Int;   t.ival=std::stol(num); t.fval=(double)t.ival; }
                t.text=num; out.push_back(t); continue;
            }
            if (std::isalpha((unsigned char)c) || c=='_') {
                std::size_t b=i;
                while(!eof() && (std::isalnum((unsigned char)peek())||peek()=='_')) get();
                out.push_back({Tok::Ident, std::string(s.substr(b,i-b)),0,0,p}); continue;
            }
            return Unexpected<Diag>{diag(std::string("unexpected char '")+c+"'", p)};
        }
        out.push_back({Tok::End,"",0,0,here()});
        return out;
    }
};

// ------------------------------- parser -------------------------------------
struct Parser {
    std::vector<Tok> t;
    std::size_t i = 0;

    const Tok& peek() const { return t[i]; }
    const Tok& adv() { return t[i < t.size()-1 ? i++ : i]; }
    SourcePos pos() const { return peek().pos; }

    bool is_ident(std::string_view w) const {
        return peek().kind==Tok::Ident && peek().text==w;
    }

    Expected<int, Diag> kv_int(std::string_view key) {
        // key = <int>
        if (!is_ident(key)) return Unexpected<Diag>{diag(std::string("expected '")+std::string(key)+"'", pos())};
        adv();
        if (peek().kind!=Tok::Eq) return Unexpected<Diag>{diag("expected '='", pos())};
        adv();
        if (peek().kind!=Tok::Int) return Unexpected<Diag>{diag("expected integer", pos())};
        return (int)adv().ival;
    }

    // $metric < value
    Expected<std::pair<std::string,double>, Diag> metric_cond() {
        if (peek().kind!=Tok::Dollar) return Unexpected<Diag>{diag("expected '$metric'", pos())};
        adv();
        if (peek().kind!=Tok::Ident) return Unexpected<Diag>{diag("expected metric name", pos())};
        std::string m = adv().text;
        if (peek().kind!=Tok::Lt) return Unexpected<Diag>{diag("expected '<'", pos())};
        adv();
        if (peek().kind!=Tok::Int && peek().kind!=Tok::Float)
            return Unexpected<Diag>{diag("expected number", pos())};
        double v = adv().fval;
        return std::pair{m, v};
    }

    Expected<std::vector<Stmt>, Diag> block() {
        if (peek().kind!=Tok::LBrace) return Unexpected<Diag>{diag("expected '{'", pos())};
        adv();
        std::vector<Stmt> body;
        while (peek().kind!=Tok::RBrace && peek().kind!=Tok::End) {
            auto st = statement();
            if (!st) return Unexpected<Diag>{st.error()};
            body.push_back(std::move(*st));
        }
        if (peek().kind!=Tok::RBrace) return Unexpected<Diag>{diag("expected '}'", pos())};
        adv();
        return body;
    }

    Expected<Stmt, Diag> statement() {
        SourcePos p = pos();
        if (peek().kind!=Tok::Ident)
            return Unexpected<Diag>{diag("expected command", p)};
        std::string cmd = peek().text;

        if (cmd=="config") {
            adv();
            auto seed = kv_int("seed");
            if (!seed) return Unexpected<Diag>{seed.error()};
            if (!is_ident("corpus")) return Unexpected<Diag>{diag("expected 'corpus'", pos())};
            adv();
            if (peek().kind!=Tok::Eq) return Unexpected<Diag>{diag("expected '='", pos())};
            adv();
            if (peek().kind!=Tok::Ident) return Unexpected<Diag>{diag("expected synthetic|real", pos())};
            std::string cv = adv().text;
            Corpus corp;
            if (cv=="synthetic") corp=Corpus::Synthetic;
            else if (cv=="real")  corp=Corpus::Real;
            else return Unexpected<Diag>{diag("corpus must be synthetic|real", p)};
            return Stmt{Config{*seed, corp, p}};
        }
        if (cmd=="eval") {
            adv(); auto n = kv_int("n");
            if (!n) return Unexpected<Diag>{n.error()};
            return Stmt{Eval{*n, p}};
        }
        if (cmd=="train") {
            adv(); auto k = kv_int("steps");
            if (!k) return Unexpected<Diag>{k.error()};
            return Stmt{Train{*k, p}};
        }
        if (cmd=="tau")  { adv(); return Stmt{Tau{p}}; }
        if (cmd=="mem")  { adv(); return Stmt{Mem{p}}; }
        if (cmd=="saddle")  { adv(); return Stmt{Saddle{p}}; }
        if (cmd=="lanczos") { adv(); return Stmt{Lanczos{p}}; }
        if (cmd=="snapper") { adv(); return Stmt{Snapper{p}}; }
        if (cmd=="topogate"){ adv(); return Stmt{TopoGate{p}}; }
        if (cmd=="tau_retry"){ adv(); return Stmt{TauRetry{p}}; }
        if (cmd=="align_lm"){ adv(); return Stmt{AlignLM{p}}; }
        if (cmd=="k0_split"){ adv(); return Stmt{K0Split{p}}; }
        if (cmd=="joint_ce"){ adv(); return Stmt{JointCE{p}}; }
        if (cmd=="basin") {
            adv();
            int ms = 150;
            if (is_ident("max")) {
                auto m = kv_int("max");
                if (!m) return Unexpected<Diag>{m.error()};
                ms = *m;
            }
            return Stmt{Basin{ms, p}};
        }
        if (cmd=="mfpump") {
            adv();
            // optional seed=<int>; default 0
            int seed = 0;
            if (is_ident("seed")) {
                auto s = kv_int("seed");
                if (!s) return Unexpected<Diag>{s.error()};
                seed = *s;
            }
            return Stmt{MfPump{seed, p}};
        }
        if (cmd=="log") {
            adv();
            if (peek().kind!=Tok::Str) return Unexpected<Diag>{diag("expected string", pos())};
            return Stmt{Log{adv().text, p}};
        }
        if (cmd=="assert") {
            adv(); auto mc = metric_cond();
            if (!mc) return Unexpected<Diag>{mc.error()};
            return Stmt{Assert{mc->first, mc->second, p}};
        }
        if (cmd=="repeat") {
            adv();
            if (peek().kind!=Tok::Int) return Unexpected<Diag>{diag("expected count", pos())};
            int cnt = (int)adv().ival;
            auto b = block();
            if (!b) return Unexpected<Diag>{b.error()};
            return Stmt{Repeat{cnt, std::move(*b), p}};
        }
        if (cmd=="until") {
            adv(); auto mc = metric_cond();
            if (!mc) return Unexpected<Diag>{mc.error()};
            if (!is_ident("max")) return Unexpected<Diag>{diag("expected 'max <int>'", pos())};
            adv();
            if (peek().kind!=Tok::Int) return Unexpected<Diag>{diag("expected max iterations", pos())};
            int mx = (int)adv().ival;
            auto b = block();
            if (!b) return Unexpected<Diag>{b.error()};
            return Stmt{Until{mc->first, mc->second, mx, std::move(*b), p}};
        }
        if (cmd=="if") {
            adv();
            If node; node.pos = p;
            auto arm_cond = [&](CondArm& arm) -> Expected<std::monostate,Diag> {
                if (peek().kind!=Tok::Dollar) return Unexpected<Diag>{diag("expected '$metric'", pos())};
                adv();
                if (peek().kind!=Tok::Ident) return Unexpected<Diag>{diag("expected metric", pos())};
                arm.metric = adv().text;
                if (peek().kind==Tok::Lt) { arm.is_lt = true; adv(); }
                else if (is_ident("ge")) { arm.is_lt = false; adv(); }
                else return Unexpected<Diag>{diag("expected '<' or 'ge'", pos())};
                if (peek().kind!=Tok::Int && peek().kind!=Tok::Float)
                    return Unexpected<Diag>{diag("expected number", pos())};
                arm.value = adv().fval;
                return std::monostate{};
            };
            {
                CondArm arm;
                if (auto c = arm_cond(arm); !c) return Unexpected<Diag>{c.error()};
                auto b = block(); if (!b) return Unexpected<Diag>{b.error()};
                arm.body = std::move(*b);
                node.arms.push_back(std::move(arm));
            }
            while (is_ident("elif")) {
                adv();
                CondArm arm;
                if (auto c = arm_cond(arm); !c) return Unexpected<Diag>{c.error()};
                auto b = block(); if (!b) return Unexpected<Diag>{b.error()};
                arm.body = std::move(*b);
                node.arms.push_back(std::move(arm));
            }
            if (is_ident("else")) {
                adv();
                CondArm arm; arm.metric = "";
                auto b = block(); if (!b) return Unexpected<Diag>{b.error()};
                arm.body = std::move(*b);
                node.arms.push_back(std::move(arm));
            }
            return Stmt{std::move(node)};
        }
        if (cmd=="pump_until") {
            // pump_until rising $<metric> [guard $<metric> == <int>] max <int> { ... }
            adv();
            if (!is_ident("rising")) return Unexpected<Diag>{diag("expected 'rising $metric'", pos())};
            adv();
            if (peek().kind!=Tok::Dollar) return Unexpected<Diag>{diag("expected '$metric'", pos())};
            adv();
            if (peek().kind!=Tok::Ident) return Unexpected<Diag>{diag("expected metric name", pos())};
            std::string rising = adv().text;
            std::string guard_m; double guard_v = 0;
            if (is_ident("guard")) {
                adv();
                if (peek().kind!=Tok::Dollar) return Unexpected<Diag>{diag("expected '$metric'", pos())};
                adv();
                if (peek().kind!=Tok::Ident) return Unexpected<Diag>{diag("expected metric name", pos())};
                guard_m = adv().text;
                // accept '==' (two Lt? no) — we tokenize '==' as... we only have '<'.
                // Use 'ge'/'==' as identifier-free: require '=' '=' via Eq Eq.
                if (peek().kind!=Tok::Eq) return Unexpected<Diag>{diag("expected '=='", pos())};
                adv();
                if (peek().kind!=Tok::Eq) return Unexpected<Diag>{diag("expected '=='", pos())};
                adv();
                if (peek().kind!=Tok::Int && peek().kind!=Tok::Float)
                    return Unexpected<Diag>{diag("expected number", pos())};
                guard_v = adv().fval;
            }
            if (!is_ident("max")) return Unexpected<Diag>{diag("expected 'max <int>'", pos())};
            adv();
            if (peek().kind!=Tok::Int) return Unexpected<Diag>{diag("expected max iterations", pos())};
            int mx = (int)adv().ival;
            auto b = block();
            if (!b) return Unexpected<Diag>{b.error()};
            return Stmt{PumpUntil{rising, guard_m, guard_v, mx, std::move(*b), p}};
        }
        return Unexpected<Diag>{diag("unknown command '"+cmd+"'", p)};
    }

    Expected<ScheduleAst, Diag> run() {
        ScheduleAst ast;
        while (peek().kind!=Tok::End) {
            auto st = statement();
            if (!st) return Unexpected<Diag>{st.error()};
            ast.stmts.push_back(std::move(*st));
        }
        return ast;
    }
};

// ------------------------------- checker ------------------------------------
// Semantic rules:
//   - config must appear exactly once, first.
//   - metric names in until/assert must be one of val/loss/tau/mem.
//   - counts/steps/iters must be positive.
bool valid_metric(const std::string& m) {
    return m=="val"||m=="loss"||m=="tau"||m=="mem"||m=="phi";
}

Expected<std::monostate, Diag> check_body(const std::vector<Stmt>& body);

Expected<std::monostate, Diag> check_stmt(const Stmt& s) {
    return std::visit([](const auto& v) -> Expected<std::monostate, Diag> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, Eval>) {
            if (v.n <= 0) return Unexpected<Diag>{diag("eval n must be > 0", v.pos)};
        } else if constexpr (std::is_same_v<T, Train>) {
            if (v.steps <= 0) return Unexpected<Diag>{diag("train steps must be > 0", v.pos)};
        } else if constexpr (std::is_same_v<T, Assert>) {
            if (!valid_metric(v.metric))
                return Unexpected<Diag>{diag("unknown metric '"+v.metric+"' (val|loss|tau|mem)", v.pos)};
        } else if constexpr (std::is_same_v<T, Repeat>) {
            if (v.count <= 0) return Unexpected<Diag>{diag("repeat count must be > 0", v.pos)};
            return check_body(v.body);
        } else if constexpr (std::is_same_v<T, Until>) {
            if (!valid_metric(v.metric))
                return Unexpected<Diag>{diag("unknown metric '"+v.metric+"'", v.pos)};
            if (v.max_iters <= 0) return Unexpected<Diag>{diag("until max must be > 0", v.pos)};
            return check_body(v.body);
        } else if constexpr (std::is_same_v<T, PumpUntil>) {
            if (!valid_metric(v.rising_metric))
                return Unexpected<Diag>{diag("unknown rising metric '"+v.rising_metric+"'", v.pos)};
            if (!v.guard_metric.empty() && !valid_metric(v.guard_metric))
                return Unexpected<Diag>{diag("unknown guard metric '"+v.guard_metric+"'", v.pos)};
            if (v.max_iters <= 0) return Unexpected<Diag>{diag("pump_until max must be > 0", v.pos)};
            return check_body(v.body);
        } else if constexpr (std::is_same_v<T, If>) {
            for (const auto& arm : v.arms) {
                if (!arm.metric.empty() && !valid_metric(arm.metric))
                    return Unexpected<Diag>{diag("unknown metric '"+arm.metric+"' in if", v.pos)};
                auto r = check_body(arm.body);
                if (!r) return r;
            }
            return std::monostate{};
        }
        return std::monostate{};
    }, s.data);
}

Expected<std::monostate, Diag> check_body(const std::vector<Stmt>& body) {
    for (const auto& s : body) {
        auto r = check_stmt(s);
        if (!r) return r;
    }
    return std::monostate{};
}

Expected<std::monostate, Diag> check(const ScheduleAst& ast) {
    if (ast.stmts.empty())
        return Unexpected<Diag>{diag("empty schedule")};
    if (!std::holds_alternative<Config>(ast.stmts.front().data))
        return Unexpected<Diag>{diag("schedule must begin with 'config'", ast.stmts.front().data.index()==0?SourcePos{}:SourcePos{})};
    for (std::size_t k=1;k<ast.stmts.size();++k)
        if (std::holds_alternative<Config>(ast.stmts[k].data))
            return Unexpected<Diag>{diag("config may appear only once")};
    return check_body(ast.stmts);
}

// ------------------------------- emitter ------------------------------------
struct Emitter {
    std::string out;
    int counter_slots = 0;
    int label_id = 0;

    std::string new_label() { return "L" + std::to_string(label_id++); }
    void line(const std::string& s) { out += s; out += '\n'; }

    void emit_body(const std::vector<Stmt>& body) {
        for (const auto& s : body) emit(s);
    }

    void emit(const Stmt& s) {
        std::visit([&](const auto& v){
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Config>)
                line("CONFIG seed=" + std::to_string(v.seed) + " corpus=" +
                     (v.corpus==Corpus::Synthetic?"synthetic":"real"));
            else if constexpr (std::is_same_v<T, Eval>)
                line("EVAL n=" + std::to_string(v.n));
            else if constexpr (std::is_same_v<T, Train>)
                line("TRAIN steps=" + std::to_string(v.steps));
            else if constexpr (std::is_same_v<T, Tau>)   line("TAU");
            else if constexpr (std::is_same_v<T, Mem>)   line("MEM");
            else if constexpr (std::is_same_v<T, Saddle>)  line("SADDLE");
            else if constexpr (std::is_same_v<T, Lanczos>) line("LANCZOS");
            else if constexpr (std::is_same_v<T, Snapper>) line("SNAPPER");
            else if constexpr (std::is_same_v<T, TopoGate>) line("TOPOGATE");
            else if constexpr (std::is_same_v<T, TauRetry>) line("TAU_RETRY");
            else if constexpr (std::is_same_v<T, AlignLM>) line("ALIGN_LM");
            else if constexpr (std::is_same_v<T, K0Split>) line("K0_SPLIT");
            else if constexpr (std::is_same_v<T, JointCE>) line("JOINT_CE");
            else if constexpr (std::is_same_v<T, Basin>)
                line("BASIN max=" + std::to_string(v.max_steps));
            else if constexpr (std::is_same_v<T, MfPump>)
                line("MFPUMP seed=" + std::to_string(v.seed));
            else if constexpr (std::is_same_v<T, Log>)   line("LOG \"" + v.message + "\"");
            else if constexpr (std::is_same_v<T, Assert>)
                line("ASSERT_LT " + v.metric + " " + std::to_string(v.value));
            else if constexpr (std::is_same_v<T, Repeat>) {
                // counter-based loop: set slot, top label, body, dec+jump-if-nonzero
                int slot = counter_slots++;
                std::string top = new_label(), end = new_label();
                line("SET_COUNTER " + std::to_string(slot) + " " + std::to_string(v.count));
                line("LABEL " + top);
                line("JZ_COUNTER " + std::to_string(slot) + " " + end);
                emit_body(v.body);
                line("DEC_COUNTER " + std::to_string(slot));
                line("JMP " + top);
                line("LABEL " + end);
            }
            else if constexpr (std::is_same_v<T, Until>) {
                // runtime-conditional loop with iteration cap
                int slot = counter_slots++;
                std::string top = new_label(), end = new_label();
                line("SET_COUNTER " + std::to_string(slot) + " " + std::to_string(v.max_iters));
                line("LABEL " + top);
                line("JZ_COUNTER " + std::to_string(slot) + " " + end);
                // if metric already satisfies (< value), exit early:
                // BRANCH_LT jumps to end when last <metric> < value
                line("BRANCH_LT " + v.metric + " " + std::to_string(v.value) + " " + end);
                emit_body(v.body);
                line("DEC_COUNTER " + std::to_string(slot));
                line("JMP " + top);
                line("LABEL " + end);
            }
            else if constexpr (std::is_same_v<T, If>) {
                std::string end = new_label();
                for (const auto& arm : v.arms) {
                    if (arm.metric.empty()) {
                        emit_body(arm.body);
                        line("JMP " + end);
                    } else {
                        std::string next = new_label();
                        // jump to `next` when the condition is FALSE (negation)
                        if (arm.is_lt)
                            line("BRANCH_GE " + arm.metric + " " +
                                 std::to_string(arm.value) + " " + next);
                        else
                            line("BRANCH_LT " + arm.metric + " " +
                                 std::to_string(arm.value) + " " + next);
                        emit_body(arm.body);
                        line("JMP " + end);
                        line("LABEL " + next);
                    }
                }
                line("LABEL " + end);
            }
            else if constexpr (std::is_same_v<T, PumpUntil>) {
                // Faithful MF-pump loop: run body up to max_iters, but stop when
                //   (a) guard metric reaches guard_value (e.g. phi == 5), OR
                //   (b) rising_metric increased for two consecutive iterations.
                // The rising check needs the metric AFTER the body runs, so the
                // guard/rising branches sit at the END of the loop body.
                int slot = counter_slots++;
                std::string top = new_label(), end = new_label();
                line("SET_COUNTER " + std::to_string(slot) + " " + std::to_string(v.max_iters));
                line("RESET_RISING " + v.rising_metric);   // clear history
                line("LABEL " + top);
                line("JZ_COUNTER " + std::to_string(slot) + " " + end);
                emit_body(v.body);
                // guard: if phi >= guard_value, done
                if (!v.guard_metric.empty())
                    line("BRANCH_GE " + v.guard_metric + " " +
                         std::to_string(v.guard_value) + " " + end);
                // rising: if rising_metric rose twice consecutively, done
                line("BRANCH_RISING2 " + v.rising_metric + " " + end);
                line("DEC_COUNTER " + std::to_string(slot));
                line("JMP " + top);
                line("LABEL " + end);
            }
        }, s.data);
    }

    std::string run(const ScheduleAst& ast) {
        for (const auto& s : ast.stmts) emit(s);
        line("HALT");
        return out;
    }
};

std::string read_file(const char* path) {
    std::ifstream f(path); std::stringstream ss; ss<<f.rdbuf(); return ss.str();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: schedc <file.sched> [out.ops]\n"); return 2; }
    std::string src = read_file(argv[1]);

    Lexer lx{src};
    auto toks = lx.run();
    if (!toks) { std::fprintf(stderr, "%s\n", toks.error().format().c_str()); return 1; }

    Parser ps{std::move(*toks)};
    auto ast = ps.run();
    if (!ast) { std::fprintf(stderr, "%s\n", ast.error().format().c_str()); return 1; }

    auto chk = check(*ast);
    if (!chk) { std::fprintf(stderr, "%s\n", chk.error().format().c_str()); return 1; }

    Emitter em;
    std::string ops = em.run(*ast);

    if (argc >= 3) {
        std::ofstream o(argv[2]); o << ops;
        std::fprintf(stderr, "wrote %s\n", argv[2]);
    } else {
        std::fputs(ops.c_str(), stdout);
    }
    return 0;
}
