#include "vgpuc/lexer.hpp"
#include <cctype>
#include <optional>
#include <charconv>

namespace vgpuc {
namespace {

// A tiny scanning cursor. Methods use "deducing this" so the same const-correct
// peek/advance work whether the cursor is const or not, without duplicated
// overloads. (Here they're non-const, but this demonstrates the pattern the
// parser leans on more heavily.)
struct Cursor {
    std::string_view s;
    std::size_t i = 0;
    std::uint32_t line = 1, col = 1;

#if defined(__cpp_explicit_this_parameter) && __cpp_explicit_this_parameter >= 202110L
    // C++23 deducing 'this': one definition serves const and non-const callers.
    template <class Self>
    char peek(this Self&& self, std::size_t k = 0) {
        std::size_t j = self.i + k;
        return j < self.s.size() ? self.s[j] : '\0';
    }
#else
    // Fallback for stdlibs/compilers without deducing 'this' (e.g. GCC < 14).
    char peek(std::size_t k = 0) const {
        std::size_t j = i + k;
        return j < s.size() ? s[j] : '\0';
    }
#endif

    bool eof() const { return i >= s.size(); }

    char get() {
        char c = s[i++];
        if (c == '\n') { ++line; col = 1; } else { ++col; }
        return c;
    }

    SourcePos here() const { return SourcePos{line, col}; }
};

bool is_ident_start(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool is_ident_cont (char c) { return std::isalnum((unsigned char)c) || c == '_'; }

std::optional<Kw> keyword_of(std::string_view w) {
    if (w == "reg")   return Kw::Reg;
    if (w == "field") return Kw::Field;
    if (w == "write") return Kw::Write;
    if (w == "poll")  return Kw::Poll;
    if (w == "kick")  return Kw::Kick;
    if (w == "ro")    return Kw::Ro;
    if (w == "wo")    return Kw::Wo;
    if (w == "rw")    return Kw::Rw;
    return std::nullopt;
}

} // namespace

Expected<std::vector<Token>, Diag> lex(std::string_view src) {
    Cursor cur{src};
    std::vector<Token> out;

    auto push = [&](TokenData d, SourcePos p) { out.push_back(Token{std::move(d), p}); };

    while (!cur.eof()) {
        char c = cur.peek();

        // whitespace
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { cur.get(); continue; }

        // line comment: // ... or # ...
        if ((c == '/' && cur.peek(1) == '/') || c == '#') {
            while (!cur.eof() && cur.peek() != '\n') cur.get();
            continue;
        }

        SourcePos start = cur.here();

        // == 
        if (c == '=' && cur.peek(1) == '=') { cur.get(); cur.get(); push(EqEq{}, start); continue; }

        // punctuation
        if (std::string_view("@{}[].,=:").find(c) != std::string_view::npos) {
            cur.get();
            push(Punct{c}, start);
            continue;
        }

        // number: decimal or 0x hex
        if (std::isdigit((unsigned char)c)) {
            std::size_t begin = cur.i;
            int base = 10;
            if (c == '0' && (cur.peek(1) == 'x' || cur.peek(1) == 'X')) {
                base = 16; cur.get(); cur.get();
                begin = cur.i;
            }
            while (!cur.eof() &&
                   (base == 16 ? std::isxdigit((unsigned char)cur.peek())
                               : std::isdigit((unsigned char)cur.peek())))
                cur.get();
            std::string_view digits = src.substr(begin, cur.i - begin);
            std::uint64_t val = 0;
            auto [ptr, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), val, base);
            if (ec != std::errc{} || ptr != digits.data() + digits.size())
                return Unexpected<Diag>{diag("invalid integer literal", start)};
            push(IntLit{val, 0}, start);
            continue;
        }

        // identifier or keyword
        if (is_ident_start(c)) {
            std::size_t begin = cur.i;
            while (!cur.eof() && is_ident_cont(cur.peek())) cur.get();
            std::string_view word = src.substr(begin, cur.i - begin);
            if (auto k = keyword_of(word)) push(Keyword{*k}, start);
            else push(Ident{std::string(word)}, start);
            continue;
        }

        return Unexpected<Diag>{diag(std::string("unexpected character '") + c + "'", start)};
    }

    push(EndOfFile{}, cur.here());
    return out;
}

} // namespace vgpuc
