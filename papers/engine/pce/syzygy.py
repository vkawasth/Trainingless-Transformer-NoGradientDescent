"""Relations and their syzygies for two-type covers (companion "Two latent types on arbitrary covers", Phase III). New module.

Stage 3.1 (triple windows): the moment map c_ij -> v_i v_j, m_T -> v_T r (|T| = 3) is monomial, so the relations among the observed
moments form the toric ideal I_A of the configuration A (columns: e_i + e_j for a pair, e_i + e_j + e_k + e_r for a triple; this is the
extended incidence matrix). Its rank gives identifiability, its integer kernel the relations, its minimal free resolution the syzygies.
Stage 3.2 (windows of size 4): m_T -> v_T (r^2 + 1) is not monomial; the ideal is computed by elimination.

  config(m, edges, triples)          the configuration matrix A ((m + 1) x n) and the coordinate names
  m2_ideal(m, edges, windows)        runs Macaulay2: minimal generators, graded and multigraded Betti numbers, the differentials
  hochster_betti(A, b, i)            dim of reduced homology H_{i-1}(Delta_b; Q), Delta_b = {F : b - A_F in NA} (Miller-Sturmfels
                                     Theorem 9.2): an independent computation of the multigraded Betti number beta_{i, b} of S/I_A
  eval_rel(expr, point)              evaluate a generator string at a point (dict name -> value)
"""
from __future__ import annotations
import itertools, os, re, subprocess, tempfile
import numpy as np


def cname(e): return "c" + "".join(map(str, e))
def mname(T): return "m" + "".join(map(str, T))


def config(m, edges, triples):
    cols, names = [], []
    for e in edges:
        a = np.zeros(m + 1, int); a[list(e)] = 1; cols.append(a); names.append(cname(e))
    for T in triples:
        a = np.zeros(m + 1, int); a[list(T)] = 1; a[m] = 1; cols.append(a); names.append(mname(T))
    return np.array(cols).T, names


def s_expr(k):
    """s_k(r) as an M2 polynomial string"""
    s = {2: "1", 3: "r"}
    for j in range(3, k): s[j + 1] = f"(r*({s[j]})+({s[j - 1]}))"
    return s[k]


def m2_ideal(m, edges, windows, multigraded=True, timeout=600):
    names, images, degs = [], [], []
    for e in edges: names.append(cname(e)); images.append(f"v_{e[0]}*v_{e[1]}"); degs.append(2)
    for T in windows:
        names.append(mname(T)); images.append("*".join(f"v_{i}" for i in T) + f"*({s_expr(len(T))})"); degs.append(len(T) + 1)
    toric = all(len(T) <= 3 for T in windows)
    if toric and multigraded:
        A, _ = config(m, edges, [T for T in windows if len(T) == 3])
        dg = "{" + ",".join("{" + ",".join(str(int(x)) for x in A[:, j]) + "}" for j in range(A.shape[1])) + "}"
    else:
        dg = "{" + ",".join(map(str, degs)) + "}"
    vs = ",".join(f"v_{i}" for i in range(m))
    script = f"""
R = QQ[{vs}, r]; S = QQ[{",".join(names)}, Degrees=>{dg}];
f = map(R, S, {{{",".join(images)}}}); I = ker f; G = mingens I;
print("GENS"); scan(flatten entries G, g -> print toString g);
print("CODIM"); print toString codim I;
C = res I; print("BETTI"); scan(1..length C, i -> print toString (i, degrees C_i));
print("DIFF"); scan(1..length C, i -> print toString entries (C.dd_i));
print("END");
"""
    with tempfile.NamedTemporaryFile("w", suffix=".m2", delete=False) as fh: fh.write(script); path = fh.name
    out = subprocess.run(["M2", "--script", path], capture_output=True, text=True, timeout=timeout).stdout; os.unlink(path)
    sec = {}; cur = None
    for line in out.splitlines():
        if line in ("GENS", "CODIM", "BETTI", "DIFF", "END"): cur = line; sec[cur] = []; continue
        if cur: sec[cur].append(line)
    betti = []
    for line in sec.get("BETTI", []):
        i, rest = line.strip("()").split(",", 1); betti.append((int(i), eval(rest.strip().replace("{", "[").replace("}", "]"))))
    return dict(names=names, gens=sec.get("GENS", []), codim=int(sec["CODIM"][0]) if sec.get("CODIM") else None, betti=betti,
                diff=sec.get("DIFF", []), toric=toric, raw=out)


def eval_rel(expr, point):
    e = expr.replace("^", "**"); return eval(e, {}, dict(point))


def model_point(m, edges, windows, v, r):
    from .covers import s_poly
    pt = {cname(e): v[e[0]] * v[e[1]] for e in edges}
    for T in windows: pt[mname(T)] = float(np.prod([v[i] for i in T])) * s_poly(len(T), r)
    return pt


# ------------------------------------------------------------------------------------------------ Hochster's formula
def in_semigroup(A, b, cache):
    """b in N A ? (A non-negative integer; depth-first with memo)"""
    key = tuple(b)
    if key in cache: return cache[key]
    if np.any(b < 0): cache[key] = False; return False
    if not np.any(b): cache[key] = True; return True
    ok = False
    for j in range(A.shape[1]):
        if np.all(A[:, j] <= b) and in_semigroup(A, b - A[:, j], cache): ok = True; break
    cache[key] = ok; return ok


def reduced_homology_ranks(faces, n):
    """faces: set of frozensets (a simplicial complex incl. the empty face). Returns {q: dim H~_q} over Q"""
    byd = {}
    for F in faces: byd.setdefault(len(F) - 1, []).append(tuple(sorted(F)))
    for d in byd: byd[d].sort()
    idx = {d: {F: k for k, F in enumerate(byd[d])} for d in byd}
    def bd(d):
        if d not in byd or d - 1 not in byd: return np.zeros((len(byd.get(d - 1, [])), len(byd.get(d, []))))
        M = np.zeros((len(byd[d - 1]), len(byd[d])))
        for k, F in enumerate(byd[d]):
            for s in range(len(F)):
                G = F[:s] + F[s + 1:]; M[idx[d - 1][G], k] = (-1) ** s
        return M
    maxd = max(byd); ranks = {d: (np.linalg.matrix_rank(bd(d)) if d in byd and d - 1 in byd and len(byd[d]) and len(byd[d - 1]) else 0) for d in range(0, maxd + 2)}
    return {q: len(byd.get(q, [])) - ranks.get(q, 0) - ranks.get(q + 1, 0) for q in range(-1, maxd + 1)}


def hochster_betti(A, b):
    """returns {i: beta_{i,b}(S/I_A)} = {i: dim H~_{i-1}(Delta_b)}"""
    n = A.shape[1]; cache = {}; faces = set()
    for k in range(0, n + 1):
        for F in itertools.combinations(range(n), k):
            if in_semigroup(A, np.asarray(b) - A[:, list(F)].sum(axis=1).astype(int) if F else np.asarray(b), cache): faces.add(frozenset(F))
    if not faces: return {}
    h = reduced_homology_ranks(faces, n); return {q + 1: h[q] for q in h if h[q]}


# ------------------------------------------------------------------------------------------------ Stage 2: closed covers, any window size
def closed_faces(windows, extra_edges=()):
    """the simplicial complex generated by the windows (observing a window observes all its marginals): faces of size >= 2"""
    F = set(tuple(e) for e in extra_edges)
    for W in windows:
        for k in range(2, len(W) + 1):
            for S in itertools.combinations(sorted(W), k): F.add(S)
    return sorted(F, key=lambda S: (len(S), S))


def fname(S): return ("c" if len(S) == 2 else "m") + "".join(map(str, S))


def m2_general(m, faces, res=True, timeout=1800):
    """ideal of m_F -> v_F s_|F|(r) over the faces, Z^m-graded by v-exponents; generators, codim, Betti numbers (total ranks and
    multidegrees), first differential"""
    names = [fname(F) for F in faces]
    images = ["*".join(f"v_{i}" for i in F) + ("" if len(F) == 2 else f"*({s_expr(len(F))})") for F in faces]
    vdeg = "{" + ",".join("{" + ",".join("1" if j == i else "0" for j in range(m)) + "}" for i in range(m)) + ",{" + ",".join(["0"] * m) + "}}"
    dg = "{" + ",".join("{" + ",".join("1" if i in F else "0" for i in range(m)) + "}" for F in faces) + "}"
    vs = ",".join(f"v_{i}" for i in range(m))
    script = f"""
R = QQ[{vs}, r, Degrees=>{vdeg}]; S = QQ[{",".join(names)}, Degrees=>{dg}];
f = map(R, S, {{{",".join(images)}}}); I = ker f; G = mingens I;
print("GENS"); scan(flatten entries G, g -> print toString g);
print("CODIM"); print toString codim I;
""" + ("""C = res I; print("RANKS"); print toString apply(1..length C, i -> rank C_i);
print("DIFF"); scan(1..min(2, length C), i -> print toString entries (C.dd_i));
""" if res else "") + 'print("END");\n'
    with tempfile.NamedTemporaryFile("w", suffix=".m2", delete=False) as fh: fh.write(script); path = fh.name
    out = subprocess.run(["M2", "--script", path], capture_output=True, text=True, timeout=timeout).stdout; os.unlink(path)
    sec = {}; cur = None
    for line in out.splitlines():
        if line in ("GENS", "CODIM", "RANKS", "DIFF", "END"): cur = line; sec[cur] = []; continue
        if cur: sec[cur].append(line)
    ranks = eval(sec["RANKS"][0].replace("{", "[").replace("}", "]")) if sec.get("RANKS") else None
    if isinstance(ranks, int): ranks = [ranks]
    return dict(names=names, gens=sec.get("GENS", []), codim=int(sec["CODIM"][0]), ranks=list(ranks) if ranks else [], diff=sec.get("DIFF", []), raw=out)


def singular_ranks(m, faces, gens, timeout=1800):
    """independent check: Singular's minimal resolution (mres) of the ideal generated by gens, with the same Z^m-grading folded to
    the weight sum_F |F| (a positive grading); returns the ranks of the free modules"""
    names = [fname(F) for F in faces]; w = ",".join(str(len(F)) for F in faces)
    script = f"""ring S = 0, ({",".join(names)}), wp({w});
ideal I = {",".join(g.replace("^", "^") for g in gens)};
resolution re = sres(std(I), 0); re = minres(re); print(betti(re), "betti"); quit;
"""
    with tempfile.NamedTemporaryFile("w", suffix=".sing", delete=False) as fh: fh.write(script); path = fh.name
    out = subprocess.run(["Singular", "-q", path], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL).stdout; os.unlink(path)
    lines = [l for l in out.splitlines() if l.strip().startswith("total:")]
    return [int(x) for x in lines[0].split(":")[1].split()] if lines else out


def model_point_faces(faces, v, r):
    from .covers import s_poly
    return {fname(F): float(np.prod([v[i] for i in F])) * (1.0 if len(F) == 2 else s_poly(len(F), r)) for F in faces}


def fibonacci_relations(faces):
    """the gluing relations c_e (m_{T u T'} - m_{T - j} m_{T' - i}) - m_T m_T' for faces T, T' with T cap T' = e = {i, j} and
    T u T' a face (|T|, |T'| >= 3)"""
    Fs = set(faces); out = []
    for T, T2 in itertools.combinations([F for F in faces if len(F) >= 3], 2):
        e = tuple(sorted(set(T) & set(T2))); U = tuple(sorted(set(T) | set(T2)))
        if len(e) != 2 or U not in Fs: continue
        i, j = e; A = tuple(x for x in T if x != j); B = tuple(x for x in T2 if x != i)
        out.append(f"{fname(e)}*({fname(U)} - {fname(A)}*{fname(B)}) - {fname(T)}*{fname(T2)}")
    return out


def split_top(r):
    out, cur, d = [], "", 0
    for ch in r:
        if ch in "({": d += 1
        if ch in ")}": d -= 1
        if ch == "," and d == 0: out.append(cur); cur = ""
        else: cur += ch
    out.append(cur); return out


def parse_entries(s):
    """an M2 'toString entries M' string -> list of rows (lists of entry strings)"""
    s = s.strip()[1:-1]; rows = []; depth = 0; cur = ""
    for ch in s:
        if ch == "{":
            depth += 1
            if depth == 1: cur = ""; continue
        if ch == "}":
            depth -= 1
            if depth == 0: rows.append(cur); continue
        cur += ch
    return [[x.strip() for x in split_top(r)] for r in rows]


def binomial_moves(gens, names):
    """for binomial generators x^u - x^w: the integer move vectors u - w (rows), via sympy"""
    import sympy
    syms = sympy.symbols(names); out = []
    for g in gens:
        P = sympy.Poly(sympy.sympify(g.replace("^", "**"), locals=dict(zip(names, syms))), *syms); terms = P.terms()
        if len(terms) != 2: raise ValueError("not a binomial: " + g)
        (e1, c1), (e2, c2) = terms; out.append(int(c1) * np.array(e1) + int(c2) * np.array(e2))           # gradient at the torus identity
    return np.array(out)
