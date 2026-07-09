#pragma once
#include <string>
#include <cstdint>
#include <format>

namespace vgpuc {

struct SourcePos {
    std::uint32_t line = 0;
    std::uint32_t col  = 0;
};

// A diagnostic carries a human message plus where it happened. This is the E
// in every std::expected<T, Diag> across the pipeline.
struct Diag {
    std::string message;
    SourcePos   where{};

    std::string format() const {
        return std::format("{}:{}: error: {}", where.line, where.col, message);
    }
};

// Convenience: build an unexpected Diag.
inline Diag diag(std::string msg, SourcePos p = {}) {
    return Diag{std::move(msg), p};
}

} // namespace vgpuc
