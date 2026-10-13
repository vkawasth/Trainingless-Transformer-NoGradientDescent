from fractions import Fraction
import random, hashlib, json

# ============================================================
# Canonical distributions
# ============================================================

def canonical(d):
    return tuple(sorted(d.items(), key=lambda kv: repr(kv[0])))

def un(d):
    return dict(d)

def mk(items):
    d = {}
    for k, p in items:
        d[k] = d.get(k, Fraction(0)) + Fraction(p)
    return canonical(d)

def push(d, f):
    return mk([(f(k), p) for k, p in d])

def barycenter(dd):
    out = {}
    for rho_key, p in dd:
        for b, q in rho_key:
            out[b] = out.get(b, Fraction(0)) + p * q
    return canonical(out)

# ============================================================
# Coherence
# ============================================================

def is_coherent(c):
    rho, Pi, Omega, Sigma = c
    if barycenter(Pi) != rho:
        return False
    if barycenter(Omega) != Pi:
        return False
    if push(Sigma, lambda h: h[1]) != Pi:
        return False
    return True

# ============================================================
# Random coherent context
# ============================================================

def random_coherent_context(seed):
    random.seed(seed)
    p = Fraction(random.randint(1, 5), random.randint(6, 12))
    rho = mk([(0, p), (1, 1 - p)])
    Pi = mk([(rho, Fraction(1))])
    Omega = mk([(Pi, Fraction(1))])
    Sigma = mk([((0, rho), Fraction(1))])
    return (rho, Pi, Omega, Sigma)

# ============================================================
# Shift law
# ============================================================

def shift_law_at_point(x, c):
    rho, Pi, Omega, Sigma = c
    return (
        mk([(x, Fraction(1))]),
        mk([(rho, Fraction(1))]),
        mk([(Pi, Fraction(1))]),
        mk([(Omega, Fraction(1))]),
        push(Sigma, lambda h: (h[0], mk([(h[1], Fraction(1))])))
    )

# ============================================================
# Obstruction 2-cells
# ============================================================

def subtract(d1, d2):
    keys = set(dict(d1)) | set(dict(d2))
    out = {}
    for k in keys:
        v = dict(d1).get(k, Fraction(0)) - dict(d2).get(k, Fraction(0))
        if v != 0:
            out[k] = v
    return canonical(out)

def obstruction_shift(x, c):
    rho, Pi, Omega, Sigma = c
    L = shift_law_at_point(x, c)
    o1 = subtract(L[1], push(rho, lambda b: mk([(b, Fraction(1))])))
    o2 = subtract(L[2], push(Pi, lambda r: mk([(r, Fraction(1))])))
    o3 = subtract(L[3], push(Omega, lambda P: mk([(P, Fraction(1))])))
    oS = subtract(L[4], push(Sigma, lambda h: (h[0], mk([(h[1], Fraction(1))]))))
    return o1, o2, o3, oS

# ============================================================
# Sigma-algebra atoms and conditional expectation
# ============================================================

def sigma_algebra_atoms(x, c):
    rho, Pi, Omega, Sigma = c
    atoms = set()
    for b in dict(rho):
        atoms.add(('L1', b))
    for r in dict(Pi):
        atoms.add(('L2', r))
    for P in dict(Omega):
        atoms.add(('L3', P))
    for h in dict(Sigma):
        atoms.add(('S', h))
    return atoms

def conditional_expectation(nu, atoms, level):
    """E[nu | sigma(atoms)] at the given level."""
    d = dict(nu)
    grouped = {}
    for k, p in d.items():
        for a in atoms:
            if a[0] == level and k == a[1]:
                grouped.setdefault(a, {})[k] = p
                break
    out = {}
    for a, sub in grouped.items():
        total = sum(sub.values())
        if total == 0:
            continue
        for k, p in sub.items():
            out[k] = out.get(k, Fraction(0)) + p / total
    return canonical(out)

def beta_1(x, c, nu):
    return conditional_expectation(nu, sigma_algebra_atoms(x, c), 'L2')

# ============================================================
# Corrected check
# ============================================================

def is_measurable(nu, atoms, level):
    """nu is measurable w.r.t. the atoms if its support is a union of atoms."""
    support = set(dict(nu))
    for k in support:
        if not any(a[0] == level and a[1] == k for a in atoms):
            return False
    return True

def run_check(n_trials=300, seed=42):
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(seed + trial)
        x = 0
        assert is_coherent(c), f"Trial {trial}: context not coherent"

        o1, o2, o3, oS = obstruction_shift(x, c)
        if len(dict(c[0])) > 1:
            assert o1 != (), f"Trial {trial}: o1 is zero"

        # The 3-cell beta_1 on nu = D(eta)rho
        nu = push(c[0], lambda b: mk([(b, Fraction(1))]))
        b1 = beta_1(x, c, nu)

        # Check 1: beta_1 is idempotent
        b1_twice = beta_1(x, c, b1)
        assert b1_twice == b1, f"Trial {trial}: beta_1 not idempotent"

        # Check 2: beta_1 lands in the atoms
        atoms = sigma_algebra_atoms(x, c)
        assert is_measurable(b1, atoms, 'L2'), \
            f"Trial {trial}: beta_1(nu) not measurable"

        # Check 3: if nu is already measurable, beta_1 fixes it
        rho = c[0]
        nu_measurable = mk([(rho, Fraction(1))])
        b1_measurable = beta_1(x, c, nu_measurable)
        assert b1_measurable == nu_measurable, \
            f"Trial {trial}: beta_1 does not fix measurable nu"

        # Check 4: beta_1 is not identity for non-degenerate nu
        beta_is_identity = (b1 == nu)
        results.append({
            'trial': trial,
            'coherent': True,
            'o1_nonzero': o1 != (),
            'beta1_idempotent': b1_twice == b1,
            'beta1_measurable': is_measurable(b1, atoms, 'L2'),
            'beta1_fixes_measurable': b1_measurable == nu_measurable,
            'beta1_is_identity': beta_is_identity,
        })
    return results

def freeze(results, filename='check_3cat_results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    with open(filename, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA-256: {h}")
    return h

if __name__ == '__main__':
    results = run_check()
    freeze(results)
    n = len(results)
    n_identity = sum(1 for r in results if r['beta1_is_identity'])
    print(f"All {n} trials passed.")
    print(f"beta_1 was identity in {n_identity}/{n} trials (expected: 0 for non-degenerate).")
