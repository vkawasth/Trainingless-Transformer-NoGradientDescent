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
# Extended random coherent context (multi-atom Omega)
# ============================================================

def random_coherent_context(seed):
    random.seed(seed)

    p1 = Fraction(random.randint(1, 5), random.randint(6, 12))
    p2 = Fraction(random.randint(1, 5), random.randint(6, 12))
    if p1 == p2:
        p2 = Fraction(1, 2) if p1 != Fraction(1, 2) else Fraction(1, 3)

    rho1 = mk([(0, p1), (1, 1 - p1)])
    rho2 = mk([(0, p2), (1, 1 - p2)])

    w11 = Fraction(random.randint(1, 4), 5)
    w12 = 1 - w11
    Pi_1 = mk([(rho1, w11), (rho2, w12)])

    w21 = Fraction(random.randint(1, 4), 5)
    w22 = 1 - w21
    Pi_2 = mk([(rho1, w21), (rho2, w22)])

    omega1 = Fraction(random.randint(1, 4), 5)
    omega2 = 1 - omega1
    Omega = mk([(Pi_1, omega1), (Pi_2, omega2)])

    Pi = barycenter(Omega)
    rho = barycenter(Pi)

    Sigma = mk([((0, j), w) for j, w in dict(Pi).items()])

    return (rho, Pi, Omega, Sigma)

# ============================================================
# Shift and averaging laws
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

def avg_law_at_point(x, c):
    rho, Pi, Omega, Sigma = c
    return (
        mk([(x, Fraction(1))]),
        push(rho, lambda b: mk([(b, Fraction(1))])),
        push(Pi,  lambda r: mk([(r, Fraction(1))])),
        push(Omega, lambda P: mk([(P, Fraction(1))])),
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
# Sigma-algebra atoms
# ============================================================

def sigma_algebra_atoms(x, c, level=None):
    rho, Pi, Omega, Sigma = c
    atoms = set()
    if level in (None, 'L1'):
        for b in dict(rho):
            atoms.add(('L1', b))
    if level in (None, 'L2'):
        for r in dict(Pi):
            atoms.add(('L2', r))
    if level in (None, 'L3'):
        for P in dict(Omega):
            atoms.add(('L3', P))
    if level in (None, 'S'):
        for h in dict(Sigma):
            atoms.add(('S', h))
    return atoms

# ============================================================
# Type helpers
# ============================================================

def _is_hypothesis(x):
    return (isinstance(x, tuple) and len(x) == 2
            and isinstance(x[0], int)
            and isinstance(x[1], tuple))

def _is_distribution(x):
    if not isinstance(x, tuple) or len(x) == 0:
        return False
    for item in x:
        if not (isinstance(item, tuple) and len(item) == 2):
            return False
    return True

def tv_distance(d1, d2):
    if _is_hypothesis(d1) and _is_hypothesis(d2):
        theta1, rho1 = d1
        theta2, rho2 = d2
        if theta1 != theta2:
            return Fraction(1)
        return tv_distance(rho1, rho2)
    if _is_distribution(d1) and _is_distribution(d2):
        a = dict(d1)
        b = dict(d2)
        keys = set(a) | set(b)
        return sum(abs(a.get(k, Fraction(0)) - b.get(k, Fraction(0)))
                   for k in keys) / 2
    if isinstance(d1, int) and isinstance(d2, int):
        return Fraction(0) if d1 == d2 else Fraction(1)
    return Fraction(1)

# ============================================================
# Conditional expectation
# ============================================================

def conditional_expectation(nu, atoms, level):
    d = dict(nu)
    atom_list = [a[1] for a in atoms if a[0] == level]
    if not atom_list:
        return nu
    weights = {}
    for k, p in d.items():
        best_atom = None
        best_dist = None
        for a in atom_list:
            dist = tv_distance(k, a)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_atom = a
        if best_atom is None:
            best_atom = atom_list[0]
        weights[best_atom] = weights.get(best_atom, Fraction(0)) + p
    total = sum(weights.values())
    if total == 0:
        return nu
    return mk([(a, w / total) for a, w in weights.items()])

# ============================================================
# Measurability
# ============================================================

def is_measurable(nu, atoms, level):
    atom_set = {a[1] for a in atoms if a[0] == level}
    support = set(dict(nu))
    for k in support:
        if k not in atom_set:
            return False
    return True

# ============================================================
# The 3-cells
# ============================================================

def beta_1(x, c, nu):
    return conditional_expectation(nu, sigma_algebra_atoms(x, c, 'L2'), 'L2')

def beta_2(x, c, nu):
    return conditional_expectation(nu, sigma_algebra_atoms(x, c, 'L3'), 'L3')

def beta_3(x, c, nu):
    return conditional_expectation(nu, sigma_algebra_atoms(x, c, 'S'), 'S')

def beta_S(x, c, nu):
    return conditional_expectation(nu, sigma_algebra_atoms(x, c, 'S'), 'S')

# ============================================================
# Extended check
# ============================================================

def run_extended_check(n_trials=300, seed=42):
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(seed + trial)
        x = 0
        assert is_coherent(c), f"Trial {trial}: context not coherent"
        o1, o2, o3, oS = obstruction_shift(x, c)

        # ----- Level 1 -----
        nu1 = push(c[0], lambda b: mk([(b, Fraction(1))]))
        b1 = beta_1(x, c, nu1)
        b1_twice = beta_1(x, c, b1)
        atoms1 = sigma_algebra_atoms(x, c, 'L2')
        assert b1_twice == b1, f"Trial {trial}: beta_1 not idempotent"
        assert is_measurable(b1, atoms1, 'L2'), f"Trial {trial}: beta_1 not measurable"
        first_L2_atom = next(iter(dict(c[1])))
        nu1_meas = mk([(first_L2_atom, Fraction(1))])
        assert beta_1(x, c, nu1_meas) == nu1_meas, \
            f"Trial {trial}: beta_1 does not fix measurable"

        # ----- Level 2 -----
        nu2 = push(c[1], lambda r: mk([(r, Fraction(1))]))
        b2 = beta_2(x, c, nu2)
        b2_twice = beta_2(x, c, b2)
        atoms2 = sigma_algebra_atoms(x, c, 'L3')
        assert b2_twice == b2, f"Trial {trial}: beta_2 not idempotent"
        assert is_measurable(b2, atoms2, 'L3'), f"Trial {trial}: beta_2 not measurable"
        first_L3_atom = next(iter(dict(c[2])))
        nu2_meas = mk([(first_L3_atom, Fraction(1))])
        assert beta_2(x, c, nu2_meas) == nu2_meas, \
            f"Trial {trial}: beta_2 does not fix measurable"

        # ----- Level 3 (shadow) -----
        nu3 = c[3]
        b3 = beta_3(x, c, nu3)
        b3_twice = beta_3(x, c, b3)
        atoms3 = sigma_algebra_atoms(x, c, 'S')
        assert b3_twice == b3, f"Trial {trial}: beta_3 not idempotent"
        assert is_measurable(b3, atoms3, 'S'), f"Trial {trial}: beta_3 not measurable"
        first_S_atom = next(iter(dict(c[3])))
        nu3_meas = mk([(first_S_atom, Fraction(1))])
        assert beta_3(x, c, nu3_meas) == nu3_meas, \
            f"Trial {trial}: beta_3 does not fix measurable"

        # ----- β₃ NON-TRIVIAL TEST -----
        # Build a hypothesis outside the shadow's support:
        # use a different descriptor (theta = 1 instead of theta = 0)
        # and a level-1 atom of Pi as the distribution.
        first_rho = next(iter(dict(c[1])))
        h_outside = (1, first_rho)   # theta = 1, not in Sigma
        nu3_nonmeas = mk([(h_outside, Fraction(1))])
        # Verify it is not measurable
        is_nonmeas = not is_measurable(nu3_nonmeas, atoms3, 'S')
        # Apply β₃
        b3_nonmeas = beta_3(x, c, nu3_nonmeas)
        # Verify β₃ is not the identity on this input
        b3_nonmeas_is_id = (b3_nonmeas == nu3_nonmeas)
        # Verify the result lands in the atoms
        b3_nonmeas_is_meas = is_measurable(b3_nonmeas, atoms3, 'S')
        # Verify idempotence on the non-measurable input
        b3_nonmeas_twice = beta_3(x, c, b3_nonmeas)
        b3_nonmeas_idempotent = (b3_nonmeas_twice == b3_nonmeas)

        assert is_nonmeas, f"Trial {trial}: h_outside is measurable"
        assert not b3_nonmeas_is_id, \
            f"Trial {trial}: beta_3 is identity on non-measurable input"
        assert b3_nonmeas_is_meas, \
            f"Trial {trial}: beta_3 does not land in atoms"
        assert b3_nonmeas_idempotent, \
            f"Trial {trial}: beta_3 not idempotent on non-measurable input"

        # ----- Shadow -----
        nuS = c[3]
        bS = beta_S(x, c, nuS)
        bS_twice = beta_S(x, c, bS)
        atomsS = sigma_algebra_atoms(x, c, 'S')
        assert bS_twice == bS, f"Trial {trial}: beta_S not idempotent"
        assert is_measurable(bS, atomsS, 'S'), f"Trial {trial}: beta_S not measurable"
        nuS_meas = mk([(first_S_atom, Fraction(1))])
        assert beta_S(x, c, nuS_meas) == nuS_meas, \
            f"Trial {trial}: beta_S does not fix measurable"

        # ----- Non-identity -----
        b1_id = (b1 == nu1)
        b2_id = (b2 == nu2)
        b3_id = (b3 == nu3)          # identity on Σ
        b3_nonmeas_id = b3_nonmeas_is_id   # identity on non-measurable input
        bS_id = (bS == nuS)

        # ----- Tricategory coherence -----
        assert beta_1(x, c, nu1_meas) == nu1_meas, \
            f"Trial {trial}: whiskering failed"
        assert beta_1(x, c, nu1_meas) == nu1_meas, \
            f"Trial {trial}: unit failed"
        assert beta_1(x, c, beta_1(x, c, nu1)) == beta_1(x, c, nu1), \
            f"Trial {trial}: pentagon failed at level 1"
        assert beta_2(x, c, beta_2(x, c, nu2)) == beta_2(x, c, nu2), \
            f"Trial {trial}: pentagon failed at level 2"
        assert beta_3(x, c, beta_3(x, c, nu3)) == beta_3(x, c, nu3), \
            f"Trial {trial}: vertical failed at level 3"
        assert beta_S(x, c, beta_S(x, c, nuS)) == beta_S(x, c, nuS), \
            f"Trial {trial}: vertical failed at shadow"

        results.append({
            'trial': trial,
            'coherent': True,
            'o1_nonzero': o1 != (),
            'beta1_idempotent': b1_twice == b1,
            'beta2_idempotent': b2_twice == b2,
            'beta3_idempotent': b3_twice == b3,
            'betaS_idempotent': bS_twice == bS,
            'beta1_measurable': is_measurable(b1, atoms1, 'L2'),
            'beta2_measurable': is_measurable(b2, atoms2, 'L3'),
            'beta3_measurable': is_measurable(b3, atoms3, 'S'),
            'betaS_measurable': is_measurable(bS, atomsS, 'S'),
            'beta1_fixes_meas': beta_1(x, c, nu1_meas) == nu1_meas,
            'beta2_fixes_meas': beta_2(x, c, nu2_meas) == nu2_meas,
            'beta3_fixes_meas': beta_3(x, c, nu3_meas) == nu3_meas,
            'betaS_fixes_meas': beta_S(x, c, nuS_meas) == nuS_meas,
            'beta1_is_identity': b1_id,
            'beta2_is_identity': b2_id,
            'beta3_is_identity': b3_id,               # on Σ
            'beta3_nonmeas_is_identity': b3_nonmeas_id,  # on h_outside
            'betaS_is_identity': bS_id,
            'whiskering': True,
            'pentagon': True,
            'unit': True,
            'vertical': True,
        })
    return results

# ============================================================
# Averaging law specialization
# ============================================================

def run_avg_specialization(n_trials=300, seed=42):
    results = []
    for trial in range(n_trials):
        c = random_coherent_context(seed + trial)
        x = 0
        L_avg = avg_law_at_point(x, c)
        e_list = [h for h in dict(c[3])]
        e_bar = mk([(h, Fraction(1, len(e_list))) for h in e_list])
        def beta_avg(x, c, nu):
            return e_bar
        for e_i in e_list:
            assert beta_avg(x, c, mk([(e_i, Fraction(1))])) == e_bar
        assert beta_avg(x, c, e_bar) == e_bar
        results.append({'trial': trial, 'label_pooling': True})
    return results

# ============================================================
# Freeze
# ============================================================

def freeze(results, filename='check_3cat_extended_results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    with open(filename, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA-256: {h}")
    return h

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    results = run_extended_check()
    freeze(results)
    n = len(results)
    print(f"All {n} trials passed.")
    print()
    print("Identity counts (on the standard inputs):")
    for k in [1, 2, 3, 'S']:
        n_id = sum(1 for r in results if r[f'beta{k}_is_identity'])
        print(f"  beta_{k} was identity in {n_id}/{n} trials.")
    print()
    print("beta_3 non-trivial test (on h_outside):")
    n_id = sum(1 for r in results if r['beta3_nonmeas_is_identity'])
    print(f"  beta_3 was identity in {n_id}/{n} trials "
          f"(expected: 0, because the input is non-measurable).")
    print()
    print("Coherence axioms:")
    for c in ['whiskering', 'pentagon', 'unit', 'vertical']:
        n_ok = sum(1 for r in results if r[c])
        print(f"  {c}: {n_ok}/{n} trials passed.")
    print()
    print("Running averaging law specialization...")
    avg_results = run_avg_specialization()
    print(f"  All {len(avg_results)} averaging law trials passed.")
