RESEARCH USE ONLY LICENSE
Version 1.0

This license governs the use of the accompanying software, data, models, or other materials (collectively, the “Materials”). By using the Materials, you agree to the terms below.

1. Permitted Use
You may use, copy, modify, and distribute the Materials solely for non-commercial research and educational purposes. Permitted uses include:

    Academic or scientific research at a nonprofit or educational institution.

    Personal, non-commercial experimentation and study.

2. Prohibited Uses
You may not use the Materials for any commercial or military purpose, including but not limited to:

    Any activity intended for or resulting in financial gain, including incorporation into a product or service sold or licensed for a fee.

    Any activity related to defense, armed forces, weapons development, surveillance, combat systems, intelligence, or military training.

    Any use by a government or private entity in support of military operations or systems.

3. No Warranty
The Materials are provided “AS IS,” without warranties of merchantability, fitness for a particular purpose, or non-infringement. The licensor assumes no liability for any damages arising from use.

4. Redistribution
If you distribute the Materials or modifications thereof, you must retain this license notice and include a clear statement that the recipient is bound by the same non-commercial and non-military restrictions.

5. Termination
Any violation of Sections 1 or 2 automatically terminates your rights under this license. Upon termination, you must cease all use and destroy copies of the Materials.

6. Governing Law
This license shall be interpreted under the laws of [Your Jurisdiction], without regard to conflict of law principles.


(fact_env) (base) vaw1@VAWs-MacBook-Pro j-holomorphic-Fukaya % python serre_approximator.py 


<img width="691" height="467" alt="Screenshot 2026-06-15 at 12 03 47 AM" src="https://github.com/user-attachments/assets/175960e1-bb87-487c-a890-caa05b9ba7f3" />

Road to Transformer Algebraic Compiler Using Fukaya Categories and AU Category

=================================================================
  SERRE CASCADE APPROXIMATOR
  6-layer student initialized from ad(J_14)^l
  Closing the crystallization gap via Kac-Moody structure
=================================================================

Stage 1: Train 24L teacher (300 steps)...
  step 100  val=3.2314  t=14s
  step 200  val=0.4628  t=32s
  step 300  val=0.2434  t=49s
  Teacher val=0.2504

Stage 2: Extract attractor Jacobian J_{L_ATT} and Serre cascade...
  J14 shape: (48, 48)  ||δJ14||=0.5925
  Level 1: ||ad(J14)^1(J_{14+1})|| = 0.2626
  Level 2: ||ad(J14)^2(J_{14+2})|| = 0.0977
  Level 3: ||ad(J14)^3(J_{14+3})|| = 0.0314
  Level 4: ||ad(J14)^4(J_{14+4})|| = 0.0179
  Level 5: ||ad(J14)^5(J_{14+5})|| = 0.0053
  Level 6: ||ad(J14)^6(J_{14+6})|| = 0.0024
  Serre cascade extracted: 6 levels

Stage 3: Build 6L Serre-initialized student...
  Serre-init 6L before fine-tune: val=3.5441

Stage 4B: 2L random + teacher embeddings (200 CE steps)...
  [2L-random] step 50  val=3.4869
  [2L-random] step 100  val=1.5893
  [2L-random] step 150  val=0.9474
  [2L-random] step 200  val=0.8917
  2L random final: val=0.8727

Stage 4C: 6L random + teacher embeddings (200 CE steps)...
  [6L-random] step 50  val=3.0464
  [6L-random] step 100  val=0.9438
  [6L-random] step 150  val=0.5548
  [6L-random] step 200  val=0.5083
  6L random final: val=0.5098

Stage 4D: 6L Serre-initialized (200 CE steps)...
  [6L-Serre] step 50  val=1.2748
  [6L-Serre] step 100  val=0.4046
  [6L-Serre] step 150  val=0.2095
  [6L-Serre] step 200  val=0.1824
  6L Serre final: val=0.1865


=================================================================
  SERRE APPROXIMATOR RESULTS
=================================================================

  Teacher (24L):                val=0.2504  params=16,057,600

  A: 2L random + emb (200 CE):  val=0.8727  params=1,617,152
  B: 6L random + emb (200 CE):  val=0.5098  params=4,242,688
  C: 6L Serre-init (0 CE):      val=3.5441  (zero-shot)
  D: 6L Serre-init (200 CE):    val=0.1865  params=4,242,688

  Crystallization gap (teacher→2L): 0.6223 nats
  Gap with 6L random:               0.2594 nats
  Gap with 6L Serre:                -0.0639 nats

  Serre init advantage over random 6L: 0.3233 nats

  READING:
  If Serre-init < random 6L:
    The cascade initialization carries genuine algebraic signal.
    The Kac-Moody structure of J14 encodes the correct subspace
    for the approximator to find the teacher's representation.

  If Serre-init ≈ random 6L:
    The cascade does not help beyond random.
    The closed-form initialization is not more informative
    than the data distribution alone.

  If 6L random >> 2L random:
    Depth (not initialization) closes the crystallization gap.
    Each additional layer adds one Serre level.
    6 layers implements Serre levels 1-6 via gradient descent.

Mean Field Init MF10 Realization

(base) vaw1@VAWs-MacBook-Pro j-holomorphic-Fukaya % python mf_extended.py 
Training teacher...
  step 100  val=3.2314
  step 200  val=0.4628
  step 300  val=0.2434
  Teacher val=0.2474

Computing v_neg...
v_neg ready.

=================================================================
EXTENDED MEAN-FIELD EXPERIMENTS
  A: Baseline (3 MF iter, lr=0.01) — confirmed val=0.024
  B: 5 MF iterations (lr=0.01)
  C: 10 MF iterations (lr=0.01)
  D: 3 MF iterations (lr=0.001, more stable)
  E: 5 MF iterations (lr=0.001)
=================================================================

  [A-MF3-lr0.01]  saddle exit: 3.5566
    MF iter 1: val=11.2438
    MF iter 2: val=9.0170
    MF iter 3: val=10.1965
    MF final: 10.0503
    settle: 0.2884
    sign: 0.2874
    basin   50: 0.0618
    basin  100: 0.0303
    basin  125: 0.0299
    basin  150: 0.0237
    basin  167: 0.0238
    FINAL=0.0215

  [B-MF5-lr0.01]  saddle exit: 3.5566
    MF iter 1: val=11.2438
    MF iter 2: val=9.0170
    MF iter 3: val=10.1965
    MF iter 4: val=9.1520
    MF iter 5: val=11.6626
    MF final: 11.5892
    settle: 0.2227
    sign: 0.2376
    basin   50: 0.0449
    basin  100: 0.0253
    basin  125: 0.0171
    basin  150: 0.0184
    basin  167: 0.0147
    FINAL=0.0168

  [C-MF10-lr0.01]  saddle exit: 3.5566
    MF iter 1: val=11.2438
    MF iter 2: val=9.0170
    MF iter 3: val=10.1965
    MF iter 4: val=9.1520
    MF iter 5: val=11.6626
    MF iter 6: val=11.6958
    MF iter 7: val=10.3805
    MF iter 8: val=10.0245
    MF iter 9: val=9.5821
    MF iter 10: val=8.3406
    MF final: 8.5087
    settle: 0.1598
    sign: 0.1540
    basin   50: 0.0300
    basin  100: 0.0276
    basin  125: 0.0216
    basin  150: 0.0182
    basin  167: 0.0119
    FINAL=0.0155

  [D-MF3-lr0.001]  saddle exit: 3.5566
    MF iter 1: val=2.4008
    MF iter 2: val=1.8519
    MF iter 3: val=1.5278
    MF final: 1.5638
    settle: 0.1731
    sign: 0.1670
    basin   50: 0.0554
    basin  100: 0.0321
    basin  125: 0.0334
    basin  150: 0.0289
    basin  167: 0.0284
    FINAL=0.0271

  [E-MF5-lr0.001]  saddle exit: 3.5566
    MF iter 1: val=2.4008
    MF iter 2: val=1.8519
    MF iter 3: val=1.5278
    MF iter 4: val=1.3851
    MF iter 5: val=1.3626
    MF final: 1.3787
    settle: 0.1498
    sign: 0.1548
    basin   50: 0.0496
    basin  100: 0.0313
    basin  125: 0.0267
    basin  150: 0.0262
    basin  167: 0.0244
    FINAL=0.0244

=================================================================
  EXTENDED MF RESULTS
=================================================================
    Teacher:      val=0.2474
    A (MF3 0.01): val=0.0215  [confirmed best]
    B (MF5 0.01): val=0.0168  diff A-B=+0.0047
    C (MF10 0.01):val=0.0155  diff A-C=+0.0060
    D (MF3 0.001):val=0.0271  diff A-D=-0.0055
    E (MF5 0.001):val=0.0244  diff A-E=-0.0029

  IF B < A or C < A: more MF iterations → deeper basin
  IF D < A: stable MF (smaller LR) finds better basin
  IF D ~ A and B ~ A: 3 iterations is optimal, LR insensitive
  

Make Corpus enter Transformer Algebraic Geometry fully aligned
(base) vaw1@VAWs-MacBook-Pro j-holomorphic-Fukaya % python mf10_analysis.py 
Training teacher...
  step 100  val=3.2314
  step 200  val=0.4628
  step 300  val=0.2434
  Teacher val=0.2474

Computing v_neg...
v_neg ready.

=================================================================
BUILDING MODELS AT DIFFERENT MF STAGES
=================================================================

  === Baseline (saddle exit only) ===
  val: 3.5756
  Computing Fisher spectrum (50 seqs)...
  Fisher lambda_1: 1.3839
  ||mean_grad||: 1.141508
  Gradient alignment with Fisher v1: -0.9806
  W_K-emb alignment (top-5 dirs): 0.0412  [baseline was 0.0456]
  Individual: ['0.025', '0.003', '0.097', '-0.078', '0.002']
  Hessian lambda_min: -4.1268  (saddle)
  ||E - E_init||: 0.3230  [baseline valley2 was 7.99]
  ||W_K - W_K_init||: 0.0000

  === MF3 (3 oscillatory iterations) ===
  val: 10.0086
  Computing Fisher spectrum (50 seqs)...
  Fisher lambda_1: 194.9983
  ||mean_grad||: 11.069562
  Gradient alignment with Fisher v1: 0.9895
  W_K-emb alignment (top-5 dirs): 0.0332  [baseline was 0.0456]
  Individual: ['0.020', '0.018', '0.035', '-0.060', '0.032']
  Hessian lambda_min: -115.3298  (saddle)
  ||E - E_init||: 56.2288  [baseline valley2 was 7.99]
  ||W_K - W_K_init||: 4.9553

  === MF10 (10 oscillatory iterations) ===
  val: 8.3052
  Computing Fisher spectrum (50 seqs)...
  Fisher lambda_1: 23.3711
  ||mean_grad||: 4.345039
  Gradient alignment with Fisher v1: -0.9443
  W_K-emb alignment (top-5 dirs): 0.0492  [baseline was 0.0456]
  Individual: ['0.015', '-0.013', '-0.028', '0.067', '-0.123']
  Hessian lambda_min: -31.7586  (saddle)
  ||E - E_init||: 106.7223  [baseline valley2 was 7.99]
  ||W_K - W_K_init||: 8.2154

=================================================================
  MF GEOMETRIC PROGRESSION
=================================================================

  Metric                           Baseline        MF3       MF10
  --------------------------------------------------------------
  val                                3.5756    10.0086     8.3052
  Fisher lambda_1                    1.3839   194.9983    23.3711
  Grad-Fisher alignment             -0.9806     0.9895    -0.9443
  W_K-emb alignment                  0.0412     0.0332     0.0492
  Hessian lambda_min                -4.1268  -115.3298   -31.7586
  ||E - E_init||                     0.3230    56.2288   106.7223
  ||W_K - W_K_init||                 0.0000     4.9553     8.2154

  INTERPRETATION:
  IF Fisher lambda_1 INCREASES with MF: 
    MF is implementing natural gradient — aligning with Fisher directions
  IF Fisher lambda_1 DECREASES:
    MF is moving to flatter regions — reducing curvature ill-conditioning
    
  IF W_K-emb alignment INCREASES:
    MF resolves the W_K-embedding misalignment (the original saddle cause)
  IF unchanged:
    MF works through a different mechanism
    
  IF Hessian lambda_min INCREASES (less negative):
    MF is driving the system toward convexity
    The remaining CE steps are in a better-conditioned landscape
    
  IF ||E-E_init|| INCREASES monotonically:
    MF is continuously moving embeddings toward valley 2
    More iterations = closer to valley 2 floor
