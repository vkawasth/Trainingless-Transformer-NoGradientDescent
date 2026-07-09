#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <optional>
#include <concepts>
#include "vgpuc/expected.hpp"
#include "vgpuc/diag.hpp"

// This project has BOTH a compile-time register model (concept-checked, like
// register_model.cpp) AND a runtime register table (because register defs come
// from parsed text). This header carries the runtime side plus the consteval
// field-packing primitives that the compile-time side would use.

namespace vgpuc {

enum class Access : std::uint8_t { RO, WO, RW };

inline bool access_readable(Access a) { return a == Access::RO || a == Access::RW; }
inline bool access_writable(Access a) { return a == Access::WO || a == Access::RW; }

inline std::string_view access_name(Access a) {
    switch (a) { case Access::RO: return "ro"; case Access::WO: return "wo"; case Access::RW: return "rw"; }
    return "?";
}

// ------- consteval field packing primitives (compile-time proofs) -----------
// Given a field [msb:lsb] in a register of `width` bits, compute mask & shift.
// consteval forces these to run at compile time when arguments are constant,
// which is how the compile-time register model validates itself. They're also
// usable at runtime for the parsed table.

struct FieldSpec {
    std::uint32_t msb;
    std::uint32_t lsb;
    constexpr std::uint32_t bits() const { return msb - lsb + 1; }
};

// consteval overflow-safe mask. If called in a constant context with bad args,
// the throw makes it a hard compile error (can't throw in constant eval).
consteval std::uint64_t field_mask_ce(FieldSpec f, std::uint32_t width) {
    if (f.lsb > f.msb || f.msb >= width) throw "invalid field bounds";
    std::uint32_t n = f.bits();
    std::uint64_t base = (n == 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << n) - 1);
    return base << f.lsb;
}

// Runtime twin (returns via expected instead of throwing).
constexpr Expected<std::uint64_t, Diag>
field_mask(FieldSpec f, std::uint32_t width) {
    if (f.lsb > f.msb || f.msb >= width)
        return Unexpected<Diag>{diag("field bounds out of range")};
    std::uint32_t n = f.bits();
    std::uint64_t base = (n == 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << n) - 1);
    return base << f.lsb;
}

constexpr std::uint64_t field_insert(std::uint64_t word, FieldSpec f,
                                     std::uint64_t v, std::uint64_t mask) {
    return (word & ~mask) | ((v << f.lsb) & mask);
}

// A runtime-known register definition, built from parsed source.
struct RegDef {
    std::string   name;
    std::uint32_t offset;
    std::uint32_t width;   // 8/16/32/64
    Access        access;
};

struct FieldDef {
    std::string   reg;     // owning register name
    std::string   name;    // field name
    FieldSpec     spec;
};

} // namespace vgpuc
