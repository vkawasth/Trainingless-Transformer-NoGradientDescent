# Study 27b — representations of event history, decision protection, attribution (DRAFT, NOT FROZEN)

Author: Vinay K Awasthi, Johns Hopkins University, vawasth1@jh.edu
Status: draft for review. Nothing in this file has been run, and no 27b outcome has been generated. Freezing means fixing every item
marked [DECIDE], writing SPEC_study27b.md, and recording its SHA-256 in FROZEN_SPEC_study27b.txt before the first run. The study is then
run once. Every deviation goes to DEVIATIONS_study27b.md; no threshold, margin, loss or sample size changes after the freeze.
Inputs fixed by earlier work: Study 26 world (study26_e2e/world26.py), Study 27a results (study27a/results_study27a.json, frozen spec
b8c45e15...), Theorem 138 (thm:relayclock), Proposition prop:outer, Theorem 28 (selection). The selection theorem plays no role here.

## 1. Three event representations
Events are signed co-revisions e = (edge, sign) on the six-source hexagon graph of Study 27a (edges (0,1),(1,2),(2,3),(3,4),(4,5),(5,0),
(0,3); the valid drawing). Two events are dependent iff their edges share a source.

| Code | Representation | Computed by | Preserves | Forgets |
|------|----------------|-------------|-----------|---------|
| R0 | full ordered history | the word itself | all order and multiplicity | nothing |
| R1 | trace | Cartier-Foata normal form (outerrelay.layers) | dependence order, multiplicity | order of independent events |
| R2 | group (braid) class | reduced RAAG normal form (outerrelay.reduce); braid key N = 2 recorded as a secondary encoding | equality in A_G | also cancelling pairs e e^-1 separated only by independent events |

Exact facts used as identities (Theorem 138, Study 27a): R0 -> R1 -> R2 are functions; activation and the relay clock T* are functions of
R1; they are not functions of R2. The braid key is checked faithful only on the ball of length <= 3 (ZA11), so R2 is defined by the RAAG
normal form, and the braid key is reported as a secondary encoding.

## 2. Hypotheses (four, pre-registered separately)
- H1 Trace compression. Dropping the order of independent events preserves decision-relevant performance: equivalence of R1 and R0 on
  unmodified histories, |E L(R1) - E L(R0)| < eps.
- H2 Braid compression. Identifying cancelling pairs as well loses information: estimate E L(R2) - E L(R1) on unmodified histories, with
  the prediction that it exceeds eps on the activation endpoint (E3 below) and lies within eps on the relay-present endpoint (E2).
- H3 Evasion robustness. On histories transformed within an equivalence class, the degradation Delta_R = L(R, H_evasion) - L(R, H) of a
  detector trained on unmodified histories is smaller for the compressed representation than for R0. Predictions:
  Delta_R1 = 0 exactly under commutation (identity); Delta_R2 = 0 exactly under commutation and safe padding (identity); the empirical
  test is Delta_R0 - Delta_R1 > eps (commutation) and Delta_R0 - Delta_R2 > eps (safe padding).
- H4 Operational safety. R2 misses activation and relay information that R1 retains: on operational padding pairs (section 4) R2 maps H
  and H' to one class while T* differs. Reported: the exact proportion of such pairs under the generative model (identity part), and the
  activation error of the R2 detector on them against the R1 detector (empirical part).
None of these predicts that a compressed representation beats R0 on unmodified histories.

## 3. Generative model of histories
- Sources 0..5; initiator 0; acting member 3 controls the coalition contexts {M,C,Y} of Study 26; flip distance D = 0.0347 (27a,
  protection witness), per-round shift delta ~ U(0.005, 0.03); protocol assumption (P) holds (no side channel).
- Honest regime: Poisson(10) co-revisions per history on uniformly chosen edges, random signs; honest reversal (correction) of a past
  event with probability 0.15 per event, inserted after Geometric(0.5) further events [DECIDE: 0.15 and 0.5].
- Coalition regime, fixed protocol: a relay chain from 0 to 3 (shortest paths, chosen uniformly), embedded in honest background events of
  the same rate; the coalition acts from its activation round.
- Coalition regime, adaptive protocol: as fixed, then the coalition applies up to k = 3 class-preserving transformations (section 4)
  chosen to minimise the score of the R0 detector trained in phase 1 (white-box on the frozen R0 model; never on R1/R2 models).
- Honest controls for false attribution: (i) common shock: every source shifts its table by the same delta in one round; (ii) honest
  cascade: a genuine update relayed 0 -> 3 along a chain, same event shape as a coalition relay, no coordinated aim; (iii) matched drift:
  honest sources drift with the same per-round delta as the coalition, without relay.
- Labels: E2 relay present (coalition vs honest, controls counted as honest); E3 activation of member 3 within the round horizon T = 12;
  T* from outerrelay.activation and flip_bound.

## 4. Transformation families (all applied by code, never by hand)
- H_comm: 20 random swaps of adjacent independent events. Preserves R1 and R2, final reports and T* (identity, checked per history).
- H_pad_safe: insert e e^-1 at a random position, accepted only if the activation map and T* are unchanged (checked per history).
  Preserves R2 and the decision trajectory; changes R0 and R1.
- H_pad_op: insert e e^-1 at a random position, kept only if activation or T* changes. Preserves R2, changes the trajectory. Used only
  for H4 (it is not an evasion: the operational truth changes).
- H_mixed: H_comm followed by H_pad_safe.
Per history, verify: final reports unchanged; for comm / pad_safe / mixed, T* and the reached set unchanged; for pad_op, R2 unchanged.

## 5. Detectors and training discipline
- One learner for all representations: L2-regularised logistic regression, C chosen by 5-fold CV on the training split over the fixed
  grid {0.01, 0.1, 1, 10}; same splits, same seed, same grid for every representation.
- Features (fixed now, not tuned later): R0, positional event indicators (edge x sign x position <= 30) plus ordered bigrams; R1, edge x
  sign x layer indicators (layer <= 30) plus layer count; R2, the same as R1 computed on the reduced word. Feature dimensions recorded.
- No change to any representation or feature map after the first model is fitted.

## 6. Decision-protection and attribution arms (model ladder)
| Model | Information | Question |
|-------|-------------|----------|
| B2 | local + outer sheaf (27a anchors) | does reconciliation protect the decision? |
| B3 | B2 + co-revision counts | do source relationships help? |
| B4 | B3 + R0 | what can complete history recover? |
| B4C | B3 + R1 | is the trace decision-sufficient? (H1) |
| B4T | B3 + R2 | is the braid class decision-sufficient? (H2, H4) |
| B7 | B2 + preselected inspection | does inspection add protection? [DECIDE: section 7] |
Primary endpoint 1, reported separately for fixed and adaptive coalitions: proportions of runs with correct decision; flag before the
flip; wrong decision without flag; undecided.
Primary endpoint 2 (never merged with endpoint 1): false culprit assignments to honest sources; false flags under common shock, cascade
and matched drift; correct identification of the corrupted set where identifiable; detected but ambiguous. Naming rule from 27a: a source
is named inconsistent only if it lies in every minimal explanation; no source is called malicious.

## 7. Inspection arm (B7) [DECIDE: one of the two]
27a ZA6: at Study 26's N = 3000 no inspection is eligible (rho = 0.0347 for each; {M,Y} would need N* = 80,544, the others 172k-349k).
- Option A (negative control / feasibility only): B7 at N = 3000; prediction: no protection; reported as a negative control, never as a
  protection mechanism.
- Option B (change the measurement design now, from 27a only): {M,Y} at N = 120,000 (about 1.5 N*), eligibility rho > 2 eta; B7 then
  tests protection, and B7 at N = 3000 remains the negative control. No other inspection is considered.

## 8. Loss, margins, sample sizes, confidence procedure [DECIDE values]
- Decision loss L: 0 correct; 1 wrong decision without flag; 0.2 flag on an honest run; 0.1 undecided; 0 flag before a coalition flip.
- Detector loss for E2 and E3: misclassification rate (primary), log loss (secondary).
- Equivalence / noninferiority margin eps = 0.02 (absolute, on misclassification and on decision loss).
- Sizes: 6,000 training and 6,000 test histories per regime (honest, fixed coalition, adaptive coalition, three controls); 2,000
  transformed test histories per family; one seed (27002).
- Confidence: paired differences on the same test histories, 95% intervals by paired bootstrap (2,000 resamples, fixed seed); H1
  equivalence if the interval lies in (-eps, eps); H2, H3 directional if the interval lies above eps; identities checked exactly.
  At the planned size the half-width is about 0.008 for a 5% disagreement rate.

## 9. Knot arm (secondary, descriptive)
Question: is a closure functional identifiable from the available marginal tables when crossing resolutions are uncertain? For the
credal set J of joint laws of unresolved crossing signs matching the recorded pairwise sign tables, report lower and upper probabilities
of each closure class (pairwise linking-number vector) and the width W_K; no resolution convention is used (J is the resolution law).
Not part of any decision model.

## 10. Freeze checklist (from the memo, mapped)
local and outer covers and restriction maps (27a, unchanged); Study 26 world and outer anchors (27a: none, {M,R,Y}, {M,H,Y}); coalition
groups ({M,C,Y} controlled); fixed and adaptive protocols (section 3); honest controls (section 3); inspection selection from 27a
(section 7); weak {M,Y} at N = 3000 as negative control; loss, margin, attribution loss, confidence (section 8); transformations and
their verification (section 4); graph, drawing, twist power N = 2, protocol (P) (sections 1, 3); knot resolution law (section 9).
Not in this study: the audit of the selection theorem, which stays a separate track.
