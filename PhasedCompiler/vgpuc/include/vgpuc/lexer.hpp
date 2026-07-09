#pragma once
#include <vector>
#include <string_view>
#include "vgpuc/token.hpp"
#include "vgpuc/expected.hpp"
#include "vgpuc/diag.hpp"

namespace vgpuc {

// Lex the whole source into a token vector, or fail with the first Diag.
// Returning Expected<vector<Token>, Diag> is the monadic entry point.
Expected<std::vector<Token>, Diag> lex(std::string_view src);

} // namespace vgpuc
