#include "vgpuc/parser.hpp"
#include <utility>

namespace vgpuc {
namespace {

struct Parser {
    std::vector<Token> toks;
    std::size_t i = 0;

    const Token& peek(std::size_t k = 0) const {
        std::size_t j = i + k;
        return j < toks.size() ? toks[j] : toks.back(); // back() is EOF
    }
    const Token& advance() { return toks[i < toks.size() ? i++ : i]; }
    SourcePos pos() const { return peek().pos; }

    bool at_kw(Kw k) const {
        if (auto* kw = peek().get_if<Keyword>()) return kw->kw == k;
        return false;
    }
    bool at_punct(char c) const {
        if (auto* p = peek().get_if<Punct>()) return p->ch == c;
        return false;
    }

    Expected<std::monostate, Diag> expect_punct(char c) {
        if (at_punct(c)) { advance(); return std::monostate{}; }
        return Unexpected<Diag>{diag(std::string("expected '") + c + "'", pos())};
    }

    Expected<std::string, Diag> expect_ident() {
        if (auto* id = peek().get_if<Ident>()) { auto n = id->name; advance(); return n; }
        return Unexpected<Diag>{diag("expected identifier", pos())};
    }

    Expected<std::uint64_t, Diag> expect_int() {
        if (auto* n = peek().get_if<IntLit>()) { auto v = n->value; advance(); return v; }
        return Unexpected<Diag>{diag("expected integer", pos())};
    }

    // reg NAME @OFFSET (ro|wo|rw) WIDTH
    Expected<Item, Diag> parse_reg() {
        SourcePos p = pos();
        advance(); // 'reg'
        return expect_ident().and_then([&](std::string name) -> Expected<Item, Diag> {
            return expect_punct('@').and_then([&](auto) {
                return expect_int();
            }).and_then([&](std::uint64_t off) -> Expected<Item, Diag> {
                Access acc;
                if      (at_kw(Kw::Ro)) acc = Access::RO;
                else if (at_kw(Kw::Wo)) acc = Access::WO;
                else if (at_kw(Kw::Rw)) acc = Access::RW;
                else return Unexpected<Diag>{diag("expected ro/wo/rw", pos())};
                advance();
                return expect_int().and_then([&](std::uint64_t w) -> Expected<Item, Diag> {
                    if (w!=8 && w!=16 && w!=32 && w!=64)
                        return Unexpected<Diag>{diag("width must be 8/16/32/64", p)};
                    return Item{RegDecl{ RegDef{ std::move(name),
                        (std::uint32_t)off, (std::uint32_t)w, acc }, p }};
                });
            });
        });
    }

    // field REG.NAME [MSB:LSB]
    Expected<Item, Diag> parse_field() {
        SourcePos p = pos();
        advance(); // 'field'
        return expect_ident().and_then([&](std::string reg) -> Expected<Item, Diag> {
            return expect_punct('.').and_then([&](auto){ return expect_ident(); })
              .and_then([&](std::string fname) -> Expected<Item, Diag> {
                return expect_punct('[').and_then([&](auto){ return expect_int(); })
                  .and_then([&](std::uint64_t msb) -> Expected<Item, Diag> {
                    return expect_punct(':').and_then([&](auto){ return expect_int(); })
                      .and_then([&](std::uint64_t lsb) -> Expected<Item, Diag> {
                        return expect_punct(']').transform([&](auto) {
                            return Item{FieldDecl{ FieldDef{ std::move(reg), std::move(fname),
                                FieldSpec{ (std::uint32_t)msb, (std::uint32_t)lsb } }, p }};
                        });
                      });
                  });
              });
        });
    }

    // write REG { f=val, f=val }
    Expected<Item, Diag> parse_write() {
        SourcePos p = pos();
        advance(); // 'write'
        auto reg = expect_ident();
        if (!reg) return Unexpected<Diag>{reg.error()};
        if (auto e = expect_punct('{'); !e) return Unexpected<Diag>{e.error()};
        std::vector<FieldAssign> assigns;
        while (!at_punct('}')) {
            SourcePos fp = pos();
            auto f = expect_ident();
            if (!f) return Unexpected<Diag>{f.error()};
            if (auto e = expect_punct('='); !e) return Unexpected<Diag>{e.error()};
            auto v = expect_int();
            if (!v) return Unexpected<Diag>{v.error()};
            assigns.push_back(FieldAssign{*f, *v, fp});
            if (at_punct(',')) advance();
            else break;
        }
        if (auto e = expect_punct('}'); !e) return Unexpected<Diag>{e.error()};
        return Item{WriteStmt{*reg, std::move(assigns), p}};
    }

    // poll REG.FIELD == VALUE
    Expected<Item, Diag> parse_poll() {
        SourcePos p = pos();
        advance(); // 'poll'
        auto reg = expect_ident();
        if (!reg) return Unexpected<Diag>{reg.error()};
        if (auto e = expect_punct('.'); !e) return Unexpected<Diag>{e.error()};
        auto f = expect_ident();
        if (!f) return Unexpected<Diag>{f.error()};
        if (!std::holds_alternative<EqEq>(peek().data))
            return Unexpected<Diag>{diag("expected '=='", pos())};
        advance();
        auto v = expect_int();
        if (!v) return Unexpected<Diag>{v.error()};
        return Item{PollStmt{*reg, *f, *v, p}};
    }

    // kick REG
    Expected<Item, Diag> parse_kick() {
        SourcePos p = pos();
        advance();
        return expect_ident().transform([&](std::string r){ return Item{KickStmt{std::move(r), p}}; });
    }

    Expected<Program, Diag> parse_program() {
        Program prog;
        while (!peek().is_eof()) {
            Expected<Item, Diag> item = Unexpected<Diag>{diag("unknown statement", pos())};
            if      (at_kw(Kw::Reg))   item = parse_reg();
            else if (at_kw(Kw::Field)) item = parse_field();
            else if (at_kw(Kw::Write)) item = parse_write();
            else if (at_kw(Kw::Poll))  item = parse_poll();
            else if (at_kw(Kw::Kick))  item = parse_kick();
            else return Unexpected<Diag>{diag("expected reg/field/write/poll/kick", pos())};

            if (!item) return Unexpected<Diag>{item.error()};
            prog.items.push_back(std::move(*item));
        }
        return prog;
    }
};

} // namespace

Expected<Program, Diag> parse(std::vector<Token> toks) {
    Parser p{std::move(toks)};
    return p.parse_program();
}

} // namespace vgpuc
