#include "vgpuc/codegen.hpp"
#include <unordered_map>
#include <bit>
#include <cstring>
#include <utility>

namespace vgpuc {
namespace {

struct Tables {
    std::unordered_map<std::string, RegDef> regs;
    // key "REG.FIELD" -> FieldDef
    std::unordered_map<std::string, FieldDef> fields;
};

std::string fkey(std::string_view reg, std::string_view f) {
    std::string k(reg); k += '.'; k += f; return k;
}

} // namespace

Expected<Compiled, Diag> analyze_and_codegen(const Program& prog) {
    Tables t;
    Compiled out;

    // Pass 1: collect declarations, reject duplicates.
    for (const auto& item : prog.items) {
        if (auto* rd = std::get_if<RegDecl>(&item)) {
            if (t.regs.contains(rd->def.name))
                return Unexpected<Diag>{diag("duplicate register '" + rd->def.name + "'", rd->pos)};
            t.regs.emplace(rd->def.name, rd->def);
            out.regs.push_back(rd->def);
        } else if (auto* fd = std::get_if<FieldDecl>(&item)) {
            auto it = t.regs.find(fd->def.reg);
            if (it == t.regs.end())
                return Unexpected<Diag>{diag("field references unknown register '" + fd->def.reg + "'", fd->pos)};
            // bounds check against the owning register width
            if (auto m = field_mask(fd->def.spec, it->second.width); !m)
                return Unexpected<Diag>{diag("field '" + fd->def.name + "' bounds out of range for "
                    + std::to_string(it->second.width) + "-bit register", fd->pos)};
            t.fields.emplace(fkey(fd->def.reg, fd->def.name), fd->def);
        }
    }

    // Pass 2: statements -> commands, with access + range enforcement.
    for (const auto& item : prog.items) {
        if (auto* w = std::get_if<WriteStmt>(&item)) {
            auto it = t.regs.find(w->reg);
            if (it == t.regs.end())
                return Unexpected<Diag>{diag("write to unknown register '" + w->reg + "'", w->pos)};
            const RegDef& reg = it->second;
            if (!access_writable(reg.access))
                return Unexpected<Diag>{diag("cannot write to " + std::string(access_name(reg.access))
                    + " register '" + w->reg + "'", w->pos)};

            std::uint64_t word = 0;
            for (const auto& a : w->assigns) {
                auto fit = t.fields.find(fkey(w->reg, a.field));
                if (fit == t.fields.end())
                    return Unexpected<Diag>{diag("unknown field '" + a.field + "' in register '" + w->reg + "'", a.pos)};
                const FieldSpec spec = fit->second.spec;
                auto mask = field_mask(spec, reg.width);
                if (!mask) return Unexpected<Diag>{mask.error()};
                std::uint64_t maxv = *mask >> spec.lsb;
                if (a.value > maxv)
                    return Unexpected<Diag>{diag("value " + std::to_string(a.value)
                        + " does not fit in " + std::to_string(spec.bits()) + "-bit field '"
                        + a.field + "'", a.pos)};
                word = field_insert(word, spec, a.value, *mask);
            }
            out.commands.push_back(Command{Op::Write, reg.offset, reg.width, word});
        } else if (auto* p = std::get_if<PollStmt>(&item)) {
            auto it = t.regs.find(p->reg);
            if (it == t.regs.end())
                return Unexpected<Diag>{diag("poll of unknown register '" + p->reg + "'", p->pos)};
            const RegDef& reg = it->second;
            if (!access_readable(reg.access))
                return Unexpected<Diag>{diag("cannot poll (read) " + std::string(access_name(reg.access))
                    + " register '" + p->reg + "'", p->pos)};
            auto fit = t.fields.find(fkey(p->reg, p->field));
            if (fit == t.fields.end())
                return Unexpected<Diag>{diag("unknown field '" + p->field + "'", p->pos)};
            // encode expected value shifted into place so the interpreter can mask+compare
            auto mask = field_mask(fit->second.spec, reg.width);
            if (!mask) return Unexpected<Diag>{mask.error()};
            std::uint64_t shifted = (p->expect << fit->second.spec.lsb) & *mask;
            out.commands.push_back(Command{Op::Poll, reg.offset, reg.width, shifted});
        } else if (auto* k = std::get_if<KickStmt>(&item)) {
            auto it = t.regs.find(k->reg);
            if (it == t.regs.end())
                return Unexpected<Diag>{diag("kick of unknown register '" + k->reg + "'", k->pos)};
            const RegDef& reg = it->second;
            if (!access_writable(reg.access))
                return Unexpected<Diag>{diag("cannot kick " + std::string(access_name(reg.access))
                    + " register '" + k->reg + "'", k->pos)};
            out.commands.push_back(Command{Op::Kick, reg.offset, reg.width, 0});
        }
    }

    return out;
}

std::vector<std::byte> serialize(const Compiled& c) {
    std::vector<std::byte> buf;
    buf.reserve(c.commands.size() * 16);
    auto put_u32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) buf.push_back(std::byte((v >> (8*i)) & 0xFF));
    };
    auto put_u64 = [&](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) buf.push_back(std::byte((v >> (8*i)) & 0xFF));
    };
    for (const auto& cmd : c.commands) {
        buf.push_back(std::byte(std::to_underlying(cmd.op)));
        buf.push_back(std::byte(cmd.width_bits / 8));
        buf.push_back(std::byte(0));
        buf.push_back(std::byte(0));
        put_u32(cmd.offset);
        put_u64(cmd.value);
    }
    return buf;
}

} // namespace vgpuc
