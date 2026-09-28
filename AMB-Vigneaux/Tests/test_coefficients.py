import itertools
import numpy as np
import pytest

G = list(itertools.product([0, 1], repeat=4))


def chsh_system(model):
    rows, rhs = [], []
    for i in (0, 1):
        for j in (0, 1):
            for x in (0, 1):
                for y in (0, 1):
                    rows.append([1 if (g[i] == x and g[2 + j] == y) else 0 for g in G])
                    rhs.append(model(i, j, x, y))
    return np.array(rows), np.array(rhs, float)


PR = lambda i, j, x, y: float((x ^ y) == (i * j))


def test_pr_box_has_signed_real_extension():
    A, b = chsh_system(PR)
    sol = np.linalg.lstsq(A.astype(float), b, rcond=None)[0]
    assert np.abs(A @ sol - b).max() < 1e-9
    assert sol.min() < -1e-9                     # necessarily signed: the PR box is contextual


def test_signalling_family_has_no_extension():
    A, b = chsh_system(lambda i, j, x, y: float(x == 0 and y == (i & j)) if i == 0 else float(x == 1 and y == 0))
    b = b.copy(); b[0] += 0.5                     # break no-signalling
    sol = np.linalg.lstsq(A.astype(float), b, rcond=None)[0]
    assert np.abs(A @ sol - b).max() > 1e-6


def test_marginal_map_unimodular_no_p_torsion():
    sp = pytest.importorskip("sympy")
    from sympy.matrices.normalforms import smith_normal_form
    A, b = chsh_system(PR)
    M = sp.Matrix(A.tolist())
    S = smith_normal_form(M, domain=sp.ZZ)
    inv = [S[k, k] for k in range(min(S.shape)) if S[k, k] != 0]
    assert set(inv) == {1}
    S2 = smith_normal_form(M.row_join(sp.Matrix(b.astype(int).tolist())), domain=sp.ZZ)
    assert [S2[k, k] for k in range(min(S2.shape)) if S2[k, k] != 0] == inv   # integer extension exists
