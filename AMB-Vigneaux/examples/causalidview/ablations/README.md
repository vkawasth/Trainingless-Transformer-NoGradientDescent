# CausalIDView: closing the Manski / IV gap to TabPFN — what was tried

Every variant is gradient-free (closed-form ridge / kernel ridge, Newton/IRLS, EM, evidence fixed points, LPs).
Numbers are endpoint RMSE against the exact oracle bounds, averaged over the first 6–10 rebuilt worlds (see each
script's `SEEDS` / range). Reference points: published logistic 0.135 / 0.175, published TabPFN-v3.5 ≈ 0.09 / ≈ 0.12.
Run any script with `CAUSALIDVIEW_DIR` set (it imports `../run.py`).

## Manski (the bounds are s − 1 and s, s(x) = P(Y = T | X))

| script | variant | Manski RMSE |
|---|---|---|
| m_constant_and_spread.py | constant s (no covariates) | 0.131 |
| (run.py) | multinomial logistic → cells → bounds | 0.132 |
| m_direct_vs_factorised.py | direct s, CV-ridge logistic | 0.106 |
| m_direct_vs_factorised.py | factorised P(T\|X)·P(Y\|T,X) | 0.107 |
| m_kernel_select.py | **linear kernel ridge, closed-form LOO ridge** (used) | **0.105** |
| m_kernel_select.py | RBF / linear+RBF / poly-2 kernels; LOO kernel selection | 0.105–0.106 |
| m_stack.py | stack of direct + factorised | 0.106 |
| m_latent_u_em.py | latent binary U, EM with IRLS M-steps | 0.109 |
| m_ard_sparse.py | ARD (sparse Bayesian, evidence fixed point) | 0.106 |
| m_ard_sparse.py | ARD-selected coordinates + kernel | 0.108 |
| m_index_directions.py | estimated low-dimensional indices (T, Y-per-arm, s directions) + kernel, 2 worlds | 0.100–0.125 vs 0.097–0.104 linear (worse) |
| m_estimated_directions.py | estimated directions, no τ | 0.133 |
| m_phd_krr.py | principal Hessian directions + KRR | 0.18 (overfits) |
| m_oracle_structure.py | **oracle** directions (not available to an estimator) | 0.04 |
| bootstrap_bias_correction.py | bagging / bootstrap bias correction (1 world) | no gain |

Reading: every feasible function class plateaus at ≈0.105 with 1024 binary labels; the oracle-direction ceiling
(0.04) says the remaining gap is in *learning the few relevant directions* from weak labels, which is where a
pretrained prior (TabPFN) helps.

## IV (Balke–Pearl monotone-IV LP on estimated cells)

| script | variant | IV RMSE | mean bias (L, U) |
|---|---|---|---|
| (run.py) | multinomial logistic | 0.169 | |
| iv_factorised_cv.py | factorised CV-ridge logistic, instrument penalised | 0.144 (8 worlds); 0.143 over 32 worlds in run_out_penalisedI_partial.txt | −0.036, +0.024 (bounds too wide) |
| iv_free_instrument.py | **same, intercept and instrument unpenalised** (used) | **0.141** (8 worlds); 0.138 over the same 32 worlds; 0.138 over 40 | −0.015, −0.024 |
| iv_cell_krr.py | multi-output kernel ridge on the 8 cells, shared+arm kernel, LOO | 0.143 | −0.016, −0.020 |
| bootstrap_bias_correction.py | bagging / bootstrap bias correction (1 world) | 0.153 / 0.162 vs raw 0.152 | |

Reading: shrinking the instrument's coefficient makes the instrument look weaker and widens the bounds; leaving that
one well-identified coefficient unpenalised removes most of the bias. What remains is variance at n = 1024.

Also kept for the record, numbers not re-reported here: m_kernels_rbf.py, m_index_spline.py (earlier variants of the kernel and index ideas above).
