#include "vgpuc/disasm.hpp"
#include <format>
#include <ranges>
#include <version>

namespace vgpuc {

namespace rv = std::ranges::views;

std::string disassemble(const Compiled& c) {
    std::string s;
    s += std::format("; {} register(s), {} command(s)\n", c.regs.size(), c.commands.size());

    // enumerate: index + command together. Feature-gated because libstdc++ 13
    // predates views::enumerate.
#if defined(__cpp_lib_ranges_enumerate) && __cpp_lib_ranges_enumerate >= 202302L
    for (auto [idx, cmd] : rv::enumerate(c.commands)) {
#else
    std::size_t idx = 0;
    for (const auto& cmd : c.commands) {
#endif
        const char* name = cmd.op == Op::Write ? "write"
                         : cmd.op == Op::Poll  ? "poll"
                         : "kick";
        s += std::format("[{:03}] {:<5} off={:#06x} w={:>2} val={:#018x}\n",
                         idx, name, cmd.offset, cmd.width_bits, cmd.value);
#if !(defined(__cpp_lib_ranges_enumerate) && __cpp_lib_ranges_enumerate >= 202302L)
        ++idx;
#endif
    }
    return s;
}

std::string hexdump(const std::vector<std::byte>& bytes) {
    std::string s;
#if defined(__cpp_lib_ranges_chunk) && __cpp_lib_ranges_chunk >= 202202L
    std::size_t row = 0;
    for (auto chunk : bytes | rv::chunk(16)) {
        s += std::format("{:08x}  ", row);
        for (std::byte b : chunk) s += std::format("{:02x} ", std::to_integer<unsigned>(b));
        s += '\n';
        row += 16;
    }
#else
    // Fallback manual chunking.
    for (std::size_t i = 0; i < bytes.size(); i += 16) {
        s += std::format("{:08x}  ", i);
        for (std::size_t j = i; j < i + 16 && j < bytes.size(); ++j)
            s += std::format("{:02x} ", std::to_integer<unsigned>(bytes[j]));
        s += '\n';
    }
#endif
    return s;
}

} // namespace vgpuc
