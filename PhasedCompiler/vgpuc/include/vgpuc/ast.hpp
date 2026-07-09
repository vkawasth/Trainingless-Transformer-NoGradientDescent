#pragma once
#include <string>
#include <vector>
#include <variant>
#include <cstdint>
#include "vgpuc/regmodel.hpp"
#include "vgpuc/diag.hpp"

namespace vgpuc {

// Declarations
struct RegDecl   { RegDef def; SourcePos pos; };
struct FieldDecl { FieldDef def; SourcePos pos; };

// Statements
struct FieldAssign { std::string field; std::uint64_t value; SourcePos pos; };
struct WriteStmt   { std::string reg; std::vector<FieldAssign> assigns; SourcePos pos; };
struct PollStmt    { std::string reg; std::string field; std::uint64_t expect; SourcePos pos; };
struct KickStmt    { std::string reg; SourcePos pos; };

using Item = std::variant<RegDecl, FieldDecl, WriteStmt, PollStmt, KickStmt>;

struct Program {
    std::vector<Item> items;
};

} // namespace vgpuc
