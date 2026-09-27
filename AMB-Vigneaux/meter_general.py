"""Contradiction meter for general binary claims ("A ⇒ B", "A ∨ ¬C", ...), exact on small covers.

Each provenance unit u asserts a conjunction of formulas over its propositions; its
context is C_u (the propositions it mentions) and its local support S_u ⊆ {0,1}^{C_u}
is the set of assignments satisfying all its formulas (the possibilistic empirical model).
The verdicts follow the database-theory / AMB correspondence already used by the gated test:

  type 0  self          S_u = ∅                                   (the unit is unsatisfiable)
  type 1  overlap       S_u and S_v have disjoint projections onto C_u ∩ C_v
  type 2  propagation   arc consistency (semijoin reduction to a fixpoint) empties a unit:
                        a chain of units forces a contradiction; can be acyclic
  type 3  cyclic        arc consistency leaves every S_u non-empty and consistent on overlaps
                        (d0 = 0), yet no global assignment exists (H^1-type, strong
                        contextuality of the reduced support presheaf). By Beeri–Fagin–
                        Maier–Yannakakis this requires the hypergraph of the core to be cyclic.

For each contradiction the meter returns a minimal unsatisfiable set of units (deletion
MUS), its type, whether its hypergraph is acyclic (GYO), and — if units carry assertion
frequencies p_u — the logical-Bell lower bound CF >= sum p_u - (k-1).
Exact: brute force over the propositions of a connected component (keep them <= ~20).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .na_contexts import gyo_reduce

Assignment = Dict[str, int]


@dataclass
class Claim:
    unit: str
    vars: Tuple[str, ...]
    pred: Callable[..., bool]          # pred(*values) -> bool, values in {0,1}
    text: str = ""
    p: float = 1.0                     # assertion frequency (sampled claims), 1 = asserted


def parse(unit: str, text: str, p: float = 1.0) -> Claim:
    """Tiny formula language, precedence low -> high:  <->   ->   |   ^   &   ~ (or !)
    with parentheses; names are identifiers.  '->' is right-associative."""
    import re
    toks = re.findall(r"<->|->|[A-Za-z_][A-Za-z_0-9]*|[~!&|^()]", text)
    names = sorted({t for t in toks if re.match(r"[A-Za-z_]", t)})
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take(t=None):
        tok = peek()
        if t is not None and tok != t:
            raise ValueError(f"expected {t!r} at {tok!r} in {text!r}")
        pos[0] += 1
        return tok

    def iff():
        a = imp()
        while peek() == "<->":
            take(); b = imp(); a = (lambda x, y: lambda e: x(e) == y(e))(a, b)
        return a

    def imp():
        a = orr()
        if peek() == "->":
            take(); b = imp(); return (lambda x, y: lambda e: (not x(e)) or y(e))(a, b)
        return a

    def orr():
        a = xor()
        while peek() == "|":
            take(); b = xor(); a = (lambda x, y: lambda e: x(e) or y(e))(a, b)
        return a

    def xor():
        a = andd()
        while peek() == "^":
            take(); b = andd(); a = (lambda x, y: lambda e: x(e) != y(e))(a, b)
        return a

    def andd():
        a = neg()
        while peek() == "&":
            take(); b = neg(); a = (lambda x, y: lambda e: x(e) and y(e))(a, b)
        return a

    def neg():
        if peek() in ("~", "!"):
            take(); a = neg(); return lambda e: not a(e)
        if peek() == "(":
            take(); a = iff(); take(")"); return a
        name = take()
        return lambda e: bool(e[name])

    f = iff()
    if peek() is not None:
        raise ValueError(f"trailing input {peek()!r} in {text!r}")

    def pred(*vals):
        return bool(f(dict(zip(names, vals))))
    return Claim(unit, tuple(names), pred, text, p)


@dataclass
class Unit:
    name: str
    vars: Tuple[str, ...]
    support: List[Tuple[int, ...]]
    p: float = 1.0


def build_units(claims: Sequence[Claim]) -> Dict[str, Unit]:
    by: Dict[str, List[Claim]] = {}
    for c in claims:
        by.setdefault(c.unit, []).append(c)
    units = {}
    for u, cs in by.items():
        vs = tuple(sorted(set(v for c in cs for v in c.vars)))
        idx = {v: i for i, v in enumerate(vs)}
        supp = [a for a in itertools.product((0, 1), repeat=len(vs))
                if all(c.pred(*[a[idx[v]] for v in c.vars]) for c in cs)]
        units[u] = Unit(u, vs, supp, min(c.p for c in cs))
    return units


def _proj(u: Unit, W: Sequence[str]):
    idx = [u.vars.index(w) for w in W]
    return {tuple(a[i] for i in idx) for a in u.support}


def arc_consistency(units: Dict[str, Unit]) -> Dict[str, Unit]:
    """Semijoin reduction to a fixpoint (pairwise consistency on overlaps)."""
    U = {k: Unit(v.name, v.vars, list(v.support), v.p) for k, v in units.items()}
    changed = True
    while changed:
        changed = False
        for a, b in itertools.permutations(U, 2):
            W = [w for w in U[a].vars if w in U[b].vars]
            if not W:
                continue
            ok = _proj(U[b], W)
            idx = [U[a].vars.index(w) for w in W]
            new = [s for s in U[a].support if tuple(s[i] for i in idx) in ok]
            if len(new) != len(U[a].support):
                U[a].support, changed = new, True
    return U


def satisfiable(units: Dict[str, Unit]) -> bool:
    vs = sorted(set(v for u in units.values() for v in u.vars))
    sets = {k: set(u.support) for k, u in units.items()}
    idx = {k: [vs.index(v) for v in u.vars] for k, u in units.items()}
    # order units to fail fast; brute force with early checks
    for a in itertools.product((0, 1), repeat=len(vs)):
        if all(tuple(a[i] for i in idx[k]) in sets[k] for k in units):
            return True
    return False


def mus(units: Dict[str, Unit]) -> List[str]:
    """Deletion-based minimal unsatisfiable subset of units (assumes the whole set is UNSAT)."""
    core = list(units)
    for k in list(core):
        trial = [c for c in core if c != k]
        if trial and not satisfiable({c: units[c] for c in trial}):
            core = trial
    return core


@dataclass
class Contradiction:
    units: List[str]
    kind: str                      # self | overlap | propagation | cyclic
    acyclic: bool
    cf_lower: Optional[float]
    texts: List[str] = field(default_factory=list)


def classify(core: List[str], units: Dict[str, Unit]) -> Tuple[str, bool]:
    sub = {k: units[k] for k in core}
    hyper = [frozenset(u.vars) for u in sub.values()]
    acyclic = len(gyo_reduce(hyper)) <= 1
    if len(core) == 1:
        return "self", acyclic
    if len(core) == 2:
        return "overlap", acyclic
    red = arc_consistency(sub)
    if any(not u.support for u in red.values()):
        return "propagation", acyclic
    return "cyclic", acyclic


def run_general(claims: Sequence[Claim], max_rounds: int = 20) -> List[Contradiction]:
    """Find disjoint-ish contradiction cores: repeatedly extract a MUS and remove one of its
    units (the least-frequent assertion first) until the remainder is satisfiable."""
    units = build_units(claims)
    texts = {}
    for c in claims:
        texts.setdefault(c.unit, []).append(c.text)
    out = []
    live = dict(units)
    for k, u in units.items():                          # self-contradictions first
        if not u.support:
            out.append(Contradiction([k], "self", True, None, texts[k]))
            live.pop(k)
    for _ in range(max_rounds):
        if satisfiable(live):
            break
        core = mus(live)
        kind, acyc = classify(core, live)
        ps = [live[k].p for k in core]
        cf = max(0.0, sum(ps) - (len(ps) - 1)) if any(p < 1 for p in ps) else None
        out.append(Contradiction(core, kind, acyc, cf, [t for k in core for t in texts[k]]))
        live.pop(min(core, key=lambda k: live[k].p))
    return out


# ---------------------------------------------------------------------------
# Gaussian fast path over F_2: units whose support is an affine subspace
# (parity / XOR / literal claims and their conjunctions). Gluing is then linear
# algebra — Gaussian elimination — instead of a sum over 2^n assignments.
# ---------------------------------------------------------------------------
def _is_affine(support: List[Tuple[int, ...]], k: int) -> bool:
    if not support:
        return True
    n = len(support)
    if n & (n - 1):
        return False
    S = set(support)
    base = support[0]
    D = [tuple(a ^ b for a, b in zip(s, base)) for s in support]
    Dset = set(D)
    return all(tuple(x ^ y for x, y in zip(d1, d2)) in Dset for d1 in D for d2 in D)


def affine_equations(u: Unit) -> Optional[List[Tuple[Dict[str, int], int]]]:
    """If S_u is an affine subspace of F_2^{C_u}, return equations {a·x = b}; else None.
    Uses the orthogonal complement of the direction space (small, per unit)."""
    k = len(u.vars)
    if not u.support:
        return [({}, 1)]                                   # 0 = 1
    if not _is_affine(u.support, k):
        return None
    base = u.support[0]
    D = [tuple(a ^ b for a, b in zip(s, base)) for s in u.support]
    eqs = []
    for a in itertools.product((0, 1), repeat=k):
        if any(a) and all(sum(x * y for x, y in zip(a, d)) % 2 == 0 for d in D):
            b = sum(x * y for x, y in zip(a, base)) % 2
            eqs.append(({v: 1 for v, ai in zip(u.vars, a) if ai}, b))
    return eqs


def f2_consistent(rows: List[Tuple[Dict[str, int], int]]) -> bool:
    """Gaussian elimination over F_2 with Python ints as bit rows: O(m · n · n/64)."""
    names = sorted({v for r, _ in rows for v in r})
    col = {v: i for i, v in enumerate(names)}
    piv: Dict[int, int] = {}                                # leading bit -> row (bits | rhs<<n)
    n = len(names)
    for r, b in rows:
        x = sum(1 << col[v] for v in r) | (b << n)
        mask = (1 << n) - 1
        while x & mask:
            lead = (x & mask).bit_length() - 1
            if lead not in piv:
                break
            x ^= piv[lead]
        if x & mask == 0:
            if x >> n:
                return False                                # 0 = 1
            continue
        piv[(x & mask).bit_length() - 1] = x
    return True


def _f2_rank(rows: List[Tuple[Dict[str, int], int]]) -> int:
    names = sorted({v for r, _ in rows for v in r})
    col = {v: i for i, v in enumerate(names)}
    piv: Dict[int, int] = {}
    for r, _ in rows:
        x = sum(1 << col[v] for v in r)
        while x:
            lead = x.bit_length() - 1
            if lead not in piv:
                piv[lead] = x
                break
            x ^= piv[lead]
    return len(piv)


def f2_certificate(rows: List[Tuple[Dict[str, int], int]]) -> Optional[Set[int]]:
    """Row indices of an inconsistent combination (y with yA = 0, yb = 1), tracked through
    elimination; None if the system is consistent. One elimination pass."""
    names = sorted({v for r, _ in rows for v in r})
    col = {v: i for i, v in enumerate(names)}
    n = len(names)
    piv: Dict[int, Tuple[int, int]] = {}
    for idx, (r, b) in enumerate(rows):
        x, tr = sum(1 << col[v] for v in r) | (b << n), 1 << idx
        mask = (1 << n) - 1
        while x & mask:
            lead = (x & mask).bit_length() - 1
            if lead not in piv:
                break
            x ^= piv[lead][0]; tr ^= piv[lead][1]
        if x & mask == 0:
            if x >> n:
                return {i for i in range(len(rows)) if (tr >> i) & 1}
            continue
        piv[(x & mask).bit_length() - 1] = (x, tr)
    return None


def run_general_f2(claims: Sequence[Claim]) -> Optional[List[Contradiction]]:
    """Fast path: if every unit is affine over F_2, find contradiction cores with
    elimination (deletion MUS, each test polynomial). Returns None if not applicable."""
    units = build_units(claims)
    eqs = {k: affine_equations(u) for k, u in units.items()}
    if any(e is None for e in eqs.values()):
        return None
    texts = {}
    for c in claims:
        texts.setdefault(c.unit, []).append(c.text)
    out, live = [], dict(eqs)
    for k in list(live):
        if not f2_consistent(live[k]):
            out.append(Contradiction([k], "self", True, None, texts[k]))
            live.pop(k)
    rows = lambda ks: [r for k in ks for r in live[k]]
    for _ in range(50):
        if f2_consistent(rows(live)):
            break
        order = [k for k in live for _ in live[k]]           # row -> unit
        cert = f2_certificate(rows(live))
        core = sorted({order[i] for i in cert}, key=list(live).index)
        core_rows = [r for k in core for r in live[k]]
        if _f2_rank(core_rows) == len(core_rows) - 1:
            pass                                              # a circuit: already minimal
        else:
          for k in list(core):                                # minimise within the certificate
            trial = [c for c in core if c != k]
            if trial and not f2_consistent(rows(trial)):
                core = trial
        kind, acyc = classify(core, {k: units[k] for k in core}) if len(core) <= 12 else ("cyclic?", False)
        ps = [units[k].p for k in core]
        cf = max(0.0, sum(ps) - (len(ps) - 1)) if any(p < 1 for p in ps) else None
        out.append(Contradiction(core, kind, acyc, cf, [t for k in core for t in texts[k]]))
        live.pop(min(core, key=lambda k: units[k].p))
    return out
