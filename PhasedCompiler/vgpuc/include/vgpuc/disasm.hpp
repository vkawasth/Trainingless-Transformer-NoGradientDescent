#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include "vgpuc/codegen.hpp"

namespace vgpuc {

// Pretty-print the compiled program (uses ranges views internally).
std::string disassemble(const Compiled& c);

// Hexdump the serialized buffer in 16-byte rows (ranges::chunk + enumerate).
std::string hexdump(const std::vector<std::byte>& bytes);

} // namespace vgpuc
