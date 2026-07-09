#pragma once
#include <string>
#include <string_view>
#include <variant>
#include <cstdint>
#include <utility>
#include "vgpuc/diag.hpp"

namespace vgpuc {

// Keyword / punctuation kinds. Using an enum with std::to_underlying practice.
enum class Kw : std::uint8_t {
    Reg, Field, Write, Poll, Kick, Ro, Wo, Rw,
};

// Token payloads as a variant. The lexer produces a stream of these.
struct Ident   { std::string name; };
struct IntLit  { std::uint64_t value; std::uint32_t width_bits; };
struct Keyword { Kw kw; };
struct Punct   { char ch; };           // @ { } [ ] . , = plus == handled below
struct EqEq    {};                     // ==
struct EndOfFile {};

using TokenData = std::variant<Ident, IntLit, Keyword, Punct, EqEq, EndOfFile>;

struct Token {
    TokenData data;
    SourcePos pos{};

    // Small helpers built on std::visit / holds_alternative.
    bool is_eof() const { return std::holds_alternative<EndOfFile>(data); }

    template <class T>
    const T* get_if() const { return std::get_if<T>(&data); }
};

inline std::string_view kw_name(Kw k) {
    switch (k) {
        case Kw::Reg:   return "reg";
        case Kw::Field: return "field";
        case Kw::Write: return "write";
        case Kw::Poll:  return "poll";
        case Kw::Kick:  return "kick";
        case Kw::Ro:    return "ro";
        case Kw::Wo:    return "wo";
        case Kw::Rw:    return "rw";
    }
    return "?";
}

} // namespace vgpuc
