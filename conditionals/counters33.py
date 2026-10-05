"""
Chamber enumeration for the 2x3 coupling family.
Exact rational arithmetic. Output: chambers, polytope types, toric surfaces.
"""
from fractions import Fraction as F
from itertools import combinations

def walls_satisfied(f1, g1, g2):
    """Return the set of wall indices that hold at (f1,g1,g2)."""
    g3 = 1 - g1 - g2
    f2 = 1 - f1
    walls = []
    # W1: f1 = g1, W2: f1 = g2, W3: f1 = g3
    # W4: f2 = g1, W5: f2 = g2, W6: f2 = g3
    if f1 == g1: walls.append(1)
    if f1 == g2: walls.append(2)
    if f1 == g3: walls.append(3)
    if f2 == g1: walls.append(4)
    if f2 == g2: walls.append(5)
    if f2 == g3: walls.append(6)
    return frozenset(walls)

def sign_pattern(f1, g1, g2):
    """Return the sign pattern (s1,...,s6) of the six linear forms."""
    g3 = 1 - g1 - g2
    f2 = 1 - f1
    return (
        sign(f1 - g1), sign(f1 - g2), sign(f1 - g3),
        sign(f2 - g1), sign(f2 - g2), sign(f2 - g3)
    )

def sign(x):
    if x > 0: return +1
    if x < 0: return -1
    return 0

def candidate_points(f1, g1, g2):
    """Return the 12 candidate vertices (intersections of pairs of facets)."""
    # Facets: x>=0, y>=0, x<=g1, y<=g2, x+y<=f1, x+y>=f1+g1+g2-1
    # In (x,y) coordinates:
    L = {
        1: ('x', 0),            # x = 0
        2: ('y', 0),            # y = 0
        3: ('x', g1),           # x = g1
        4: ('y', g2),           # y = g2
        5: ('sum', f1),         # x+y = f1
        6: ('sum', f1+g1+g2-1), # x+y = f1+g1+g2-1
    }
    pts = []
    for i, j in combinations([1,2,3,4,5,6], 2):
        pt = intersect(L[i], L[j])
        if pt is not None:
            pts.append((i, j, pt))
    return pts

def intersect(l1, l2):
    """Intersect two lines given as (type, value). Return (x,y) or None."""
    # Lines: x=a, y=b, x+y=c
    t1, v1 = l1
    t2, v2 = l2
    if t1 == t2:
        return None  # parallel or coincident
    if {t1, t2} == {'x', 'y'}:
        return (v1, v2)
    if {t1, t2} == {'x', 'sum'}:
        return (v1, v2 - v1)
    if {t1, t2} == {'y', 'sum'}:
        return (v2 - v1, v1)
    return None

def is_feasible(pt, f1, g1, g2, eps=F(0)):
    """Check whether (x,y) satisfies all six inequalities."""
    x, y = pt
    if x < -eps: return False
    if y < -eps: return False
    if x > g1 + eps: return False
    if y > g2 + eps: return False
    if x + y > f1 + eps: return False
    if x + y < f1 + g1 + g2 - 1 - eps: return False
    return True

def vertices(f1, g1, g2):
    """Return the vertices of Cpl(f,g) in cyclic order."""
    pts = []
    for i, j, pt in candidate_points(f1, g1, g2):
        if is_feasible(pt, f1, g1, g2):
            pts.append((pt, i, j))
    # Deduplicate
    seen = set()
    unique = []
    for pt, i, j in pts:
        if pt not in seen:
            seen.add(pt)
            unique.append((pt, i, j))
    # Sort cyclically around centroid
    if not unique:
        return []
    cx = sum(p[0][0] for p in unique) / len(unique)
    cy = sum(p[0][1] for p in unique) / len(unique)
    unique.sort(key=lambda p: atan2(p[0][1]-cy, p[0][0]-cx))
    return unique

def active_facets(f1, g1, g2, vertices):
    """Return the set of facets that are active (contain an edge)."""
    active = set()
    n = len(vertices)
    for k in range(n):
        i1, j1, _ = vertices[k][1], vertices[k][2], vertices[k][0]
        i2, j2, _ = vertices[(k+1)%n][1], vertices[(k+1)%n][2], vertices[(k+1)%n][0]
        # The edge between consecutive vertices is the intersection of the
        # facet that is common to both pairs
        common = set([i1, j1]) & set([i2, j2])
        active |= common
    return active

def facet_normal(facet, g1, g2):
    """Return the outward primitive normal of a facet."""
    if facet == 1: return (-1, 0)   # x >= 0
    if facet == 2: return (0, -1)   # y >= 0
    if facet == 3: return (1, 0)    # x <= g1
    if facet == 4: return (0, 1)    # y <= g2
    if facet == 5: return (1, 1)    # x+y <= f1
    if facet == 6: return (-1, -1)  # x+y >= f1+g1+g2-1

def is_delzant(f1, g1, g2):
    """Check Delzant: at each vertex, the two active facet normals form a Z-basis."""
    vs = vertices(f1, g1, g2)
    if not vs: return False
    n = len(vs)
    for k in range(n):
        i1, j1 = vs[k][1], vs[k][2]
        i2, j2 = vs[(k+1)%n][1], vs[(k+1)%n][2]
        common = set([i1, j1]) & set([i2, j2])
        if len(common) != 1:
            return False
        # The two facets at this vertex are the two that are NOT common
        # Actually: vertex is intersection of two facets; the two facets are
        # the pair that intersect there
        # We need the two facets at the vertex
        facets_at_v = list(set([i1, j1]))  # both facets pass through vertex
        if len(facets_at_v) != 2: return False
        n1 = facet_normal(facets_at_v[0], g1, g2)
        n2 = facet_normal(facets_at_v[1], g1, g2)
        det = n1[0]*n2[1] - n1[1]*n2[0]
        if abs(det) != 1:
            return False
    return True

def chamber_key(f1, g1, g2):
    """Return a canonical key for the chamber containing (f1,g1,g2)."""
    return sign_pattern(f1, g1, g2)

def enumerate_chambers():
    """Enumerate all chambers by sampling the sign patterns."""
    # We sample on a fine grid and collect sign patterns
    patterns = set()
    N = 60
    for i in range(1, N):
        f1 = F(i, N)
        for j in range(1, N):
            g1 = F(j, N)
            for k in range(1, N):
                g2 = F(k, N)
                if g1 + g2 >= 1: continue
                if f1 == g1 or f1 == g2 or f1 == 1-g1-g2: continue
                if 1-f1 == g1 or 1-f1 == g2 or 1-f1 == 1-g1-g2: continue
                patterns.add(chamber_key(f1, g1, g2))
    return patterns

def representative(pat):
    """Find a representative point for a sign pattern."""
    # Search on a grid
    N = 120
    for i in range(1, N):
        f1 = F(i, N)
        for j in range(1, N):
            g1 = F(j, N)
            for k in range(1, N):
                g2 = F(k, N)
                if g1 + g2 >= 1: continue
                if sign_pattern(f1, g1, g2) == pat:
                    return (f1, g1, g2)
    return None

def polytope_type(f1, g1, g2):
    """Return the number of vertices and facets."""
    vs = vertices(f1, g1, g2)
    if not vs:
        return 0, 0
    facets = active_facets(f1, g1, g2, vs)
    return len(vs), len(facets)

def toric_surface_signature(f1, g1, g2):
    """Return the self-intersection numbers of the toric divisors."""
    vs = vertices(f1, g1, g2)
    facets = sorted(active_facets(f1, g1, g2, vs))
    # Compute self-intersections via wall relations
    # For a smooth complete fan with primitive rays v_i in cyclic order,
    # v_{i-1} + v_{i+1} = -a_i v_i, and D_i^2 = -a_i
    normals = [facet_normal(f, g1, g2) for f in facets]
    # Sort normals cyclically
    n = len(normals)
    if n == 0: return []
    cx = sum(v[0] for v in normals) / n
    cy = sum(v[1] for v in normals) / n
    normals_sorted = sorted(normals, key=lambda v: atan2(v[1]-cy, v[0]-cx))
    # For each ray, find a_i such that v_{i-1} + v_{i+1} = -a_i v_i
    self_int = []
    for i in range(n):
        v_prev = normals_sorted[(i-1) % n]
        v_next = normals_sorted[(i+1) % n]
        v_i = normals_sorted[i]
        # Solve v_prev + v_next = -a * v_i
        # Try a = 1, 2, 3, ...
        found = False
        for a in range(1, 10):
            if (v_prev[0] + v_next[0] == -a * v_i[0] and
                v_prev[1] + v_next[1] == -a * v_i[1]):
                self_int.append(-a)
                found = True
                break
        if not found:
            self_int.append(None)
    return list(zip(facets, normals_sorted, self_int))

if __name__ == "__main__":
    from math import atan2
    
    # Enumerate chambers
    patterns = enumerate_chambers()
    print(f"Number of 3D chambers: {len(patterns)}")
    
    # For each pattern, find a representative and compute polytope type
    results = []
    for pat in sorted(patterns):
        rep = representative(pat)
        if rep is None:
            print(f"  No representative for {pat}")
            continue
        f1, g1, g2 = rep
        nv, nf = polytope_type(f1, g1, g2)
        delz = is_delzant(f1, g1, g2)
        results.append((pat, rep, nv, nf, delz))
    
    print(f"\n{'Pattern':<20} {'Rep (f1,g1,g2)':<25} {'#V':<4} {'#F':<4} {'Delzant'}")
    print("-" * 70)
    for pat, rep, nv, nf, delz in results:
        rep_str = f"({rep[0]},{rep[1]},{rep[2]})"
        print(f"{str(pat):<20} {rep_str:<25} {nv:<4} {nf:<4} {delz}")
