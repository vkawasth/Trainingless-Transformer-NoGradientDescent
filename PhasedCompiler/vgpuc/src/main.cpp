#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <version>

#include "vgpuc/lexer.hpp"
#include "vgpuc/parser.hpp"
#include "vgpuc/codegen.hpp"
#include "vgpuc/disasm.hpp"

// Prefer std::print; fall back to fputs for older stdlibs.
#if defined(__cpp_lib_print) && __cpp_lib_print >= 202207L
  #include <print>
  #define OUT(...)  std::print(__VA_ARGS__)
  #define OUTLN(...) std::println(__VA_ARGS__)
#else
  #include <format>
  #define OUT(...)  std::fputs(std::format(__VA_ARGS__).c_str(), stdout)
  #define OUTLN(...) (std::fputs(std::format(__VA_ARGS__).c_str(), stdout), std::fputc('\n', stdout))
#endif

using namespace vgpuc;

static std::string read_all(const char* path) {
    std::ifstream f(path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) { OUTLN("usage: vgpuc <file.vg>"); return 2; }
    std::string src = read_all(argv[1]);

    // The whole front-end as one monadic pipeline: lex -> parse -> analyze.
    // Each stage returns Expected<_, Diag>; and_then threads success, short-
    // circuits on the first error.
    auto result = lex(src)
        .and_then(parse)
        .and_then(analyze_and_codegen);

    if (!result) {
        OUTLN("{}", result.error().format());
        return 1;
    }

    const Compiled& compiled = *result;
    auto bytes = serialize(compiled);

    OUT("{}", disassemble(compiled));
    OUTLN("--- {} bytes ---", bytes.size());
    OUT("{}", hexdump(bytes));
    return 0;
}
