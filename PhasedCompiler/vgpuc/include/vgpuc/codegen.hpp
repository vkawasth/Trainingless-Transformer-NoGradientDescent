#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>
#include "vgpuc/ast.hpp"
#include "vgpuc/regmodel.hpp"
#include "vgpuc/expected.hpp"

namespace vgpuc {

// Opcodes for the emitted command buffer.
enum class Op : std::uint8_t { Write = 1, Poll = 2, Kick = 3 };

// A resolved, checked command ready to serialize. Fixed 16-byte record:
//   [0]   op
//   [1]   width_bytes
//   [2..3] reserved
//   [4..7] offset (LE u32)
//   [8..15] value/expected (LE u64)
struct Command {
    Op            op;
    std::uint32_t offset;
    std::uint32_t width_bits;
    std::uint64_t value;
};

struct Compiled {
    std::vector<RegDef> regs;
    std::vector<Command> commands;
};

// Semantic pass + codegen combined: resolves names, enforces access policy
// (write to RO -> error) and field bounds, packs field assignments into a
// single register value using the consteval-twin packing helpers.
Expected<Compiled, Diag> analyze_and_codegen(const Program& prog);

// Serialize to the packed byte buffer (16 bytes per command).
std::vector<std::byte> serialize(const Compiled& c);

} // namespace vgpuc
