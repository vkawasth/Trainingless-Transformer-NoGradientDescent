#pragma once
#include "vgpuc/ast.hpp"
#include "vgpuc/token.hpp"
#include "vgpuc/expected.hpp"

namespace vgpuc {

Expected<Program, Diag> parse(std::vector<Token> toks);

} // namespace vgpuc
