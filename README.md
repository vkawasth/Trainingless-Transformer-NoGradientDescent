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
