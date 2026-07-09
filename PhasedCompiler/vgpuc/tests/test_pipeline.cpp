// Minimal dependency-free test harness. Each test returns bool; main tallies.
#include <cstdio>
#include <string>
#include <string_view>

#include "vgpuc/lexer.hpp"
#include "vgpuc/parser.hpp"
#include "vgpuc/codegen.hpp"

using namespace vgpuc;

static int g_failures = 0;
#define CHECK(cond) do { if(!(cond)){ std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while(0)

static Expected<Compiled, Diag> compile(std::string_view src) {
    return lex(src).and_then(parse).and_then(analyze_and_codegen);
}

static void test_happy_path() {
    std::printf("test_happy_path\n");
    auto r = compile(
        "reg CTRL @0x00 rw 32\n"
        "field CTRL.enable [0:0]\n"
        "field CTRL.qid [11:4]\n"
        "write CTRL { enable=1, qid=0x2A }\n");
    CHECK(r.has_value());
    if (!r) { std::printf("  err: %s\n", r.error().format().c_str()); return; }
    CHECK(r->commands.size() == 1);
    // enable=1 (bit0) | qid=0x2A<<4 => 0x2A1
    CHECK(r->commands[0].value == 0x2A1);
    CHECK(r->commands[0].op == Op::Write);
}

static void test_write_to_ro_rejected() {
    std::printf("test_write_to_ro_rejected\n");
    auto r = compile(
        "reg STATUS @0x04 ro 32\n"
        "field STATUS.ready [0:0]\n"
        "write STATUS { ready=1 }\n");
    CHECK(!r.has_value());
    if (!r) CHECK(r.error().message.find("cannot write") != std::string::npos);
}

static void test_field_overflow_rejected() {
    std::printf("test_field_overflow_rejected\n");
    auto r = compile(
        "reg CTRL @0x00 rw 32\n"
        "field CTRL.qid [11:4]\n"     // 8-bit field, max 255
        "write CTRL { qid=0x100 }\n"); // 256 -> too wide
    CHECK(!r.has_value());
    if (!r) CHECK(r.error().message.find("does not fit") != std::string::npos);
}

static void test_field_out_of_range_rejected() {
    std::printf("test_field_out_of_range_rejected\n");
    auto r = compile(
        "reg SMALL @0x00 rw 8\n"
        "field SMALL.hi [40:36]\n");  // beyond 8-bit register
    CHECK(!r.has_value());
}

static void test_poll_on_wo_rejected() {
    std::printf("test_poll_on_wo_rejected\n");
    auto r = compile(
        "reg DB @0x08 wo 32\n"
        "field DB.x [0:0]\n"
        "poll DB.x == 1\n");
    CHECK(!r.has_value());
}

static void test_lex_error_propagates() {
    std::printf("test_lex_error_propagates\n");
    auto r = compile("reg CTRL @0x00 rw 32\n$$$\n");
    CHECK(!r.has_value());
}

int main() {
    test_happy_path();
    test_write_to_ro_rejected();
    test_field_overflow_rejected();
    test_field_out_of_range_rejected();
    test_poll_on_wo_rejected();
    test_lex_error_propagates();

    if (g_failures == 0) { std::printf("\nALL TESTS PASSED\n"); return 0; }
    std::printf("\n%d CHECK(s) FAILED\n", g_failures);
    return 1;
}
