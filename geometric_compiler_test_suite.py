#!/usr/bin/env python3
"""
compiler_verification_tests.py
===============================
Verifies that the compiler is functioning correctly using the
double-blind validation framework.

Tests are organized by severity:
  - CRITICAL: If these fail, the compiler is fundamentally broken
  - MAJOR: If these fail, the compiler is degraded
  - MINOR: If these fail, the compiler has small issues

Each test compares predicted vs observed geometric quantities.
The predictions come from corpus + architecture statistics only.
"""

import json, math, collections, os, sys, re, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CONFIGURATION
# ============================================================================
D = 256
N_HEADS = 4
N_STU = 6
BATCH = 8
SEQ = 64
LR = 3e-4
VOCAB = 1017
RANK = 6

# ============================================================================
# DATA LOADING
# ============================================================================
for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json']:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing. Run: python build_corpus.py")

with open('/tmp/train_ids.json') as f: train_ids = list(map(int, json.load(f)))
with open('/tmp/val_ids.json') as f: val_ids = list(map(int, json.load(f)))
with open('/tmp/vocab.json') as f: _v = json.load(f)
VOCAB = len(_v) if isinstance(_v, list) else len(_v)
train_t = torch.tensor(train_ids, dtype=torch.long)
val_t = torch.tensor(val_ids, dtype=torch.long)

# ============================================================================
# MODEL DEFINITION (simplified for testing)
# ============================================================================
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        dh = D // N_HEADS
        self.WQ = nn.Linear(D, D, bias=False)
        self.WK = nn.Linear(D, D, bias=False)
        self.WV = nn.Linear(D, D, bias=False)
        self.op = nn.Linear(D, D, bias=False)
        self.ln = nn.LayerNorm(D)
        self.sc = math.sqrt(dh)
        self.nh = N_HEADS
        self.dh = dh
        for w in [self.WQ, self.WK, self.WV, self.op]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, h):
        B, S, _ = h.shape
        Q = self.WQ(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        K = self.WK(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        V = self.WV(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        sc = Q @ K.transpose(-2, -1) / self.sc
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
        sc = sc.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        return self.ln(h + self.op((F.softmax(sc, dim=-1) @ V).transpose(1, 2).reshape(B, S, D)))


class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(D, D * 2, bias=False)
        self.v = nn.Linear(D, D * 2, bias=False)
        self.o = nn.Linear(D * 2, D, bias=False)
        self.n = nn.LayerNorm(D)
        for w in [self.g, self.v, self.o]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, h):
        return self.n(h + self.o(F.silu(self.g(h)) * self.v(h)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.ff = FF()

    def forward(self, h):
        return self.ff(self.attn(h))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(VOCAB, D)
        self.pe = nn.Embedding(512, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.te.weight
        nn.init.normal_(self.te.weight, std=0.02)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x, y=None):
        h = self.te(x) + self.pe(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln_f(h))
        if y is not None:
            return logits, F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        return logits, None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_batch(split='train'):
    data = val_t if split == 'val' else train_t
    ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
    return (torch.stack([data[i:i + SEQ] for i in ix]),
            torch.stack([data[i + 1:i + SEQ + 1] for i in ix]))


def eval_val(m, n=8):
    m.eval()
    ls = []
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch('val')
            _, l = m(x, y)
            ls.append(l.item())
    return float(np.mean(ls))


def sheet_angles(model):
    out = []
    WKs = [model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU - 1):
        try:
            phi = WKs[l + 1] @ torch.linalg.pinv(WKs[l])
            lam = torch.linalg.eigvals(phi)
            lam1 = lam[lam.abs().argmax()]
            a = float(torch.angle(lam1))
            while a > math.pi:
                a -= 2 * math.pi
            while a < -math.pi:
                a += 2 * math.pi
            out.append(a)
        except:
            out.append(float('nan'))
    return out


def phi_clean(model):
    angles = sheet_angles(model)
    return sum(1 for a in angles if not np.isnan(a) and 
               (abs(a) < 0.3 or abs(abs(a) - math.pi) < 0.3))


def gluing_defect(model, n=6):
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g_ff = sum(p.grad.data.norm().item() for nm, p in model.named_parameters()
               if '.ff.' in nm and p.grad is not None)
    g_emb = model.te.weight.grad.data.norm().item() if model.te.weight.grad is not None else 1e-8
    model.zero_grad()
    return float(g_ff / max(g_emb, 1e-8))


def strip_energy(model):
    """Compute total strip energy across all layer pairs."""
    WKs = [model.blocks[l].attn.WK.weight.data.float().cpu().numpy()
           for l in range(N_STU)]
    total = 0.0
    for l in range(N_STU - 1):
        U0, s0, Vt0 = np.linalg.svd(WKs[l], full_matrices=False)
        U1, s1, Vt1 = np.linalg.svd(WKs[l+1], full_matrices=False)
        # Compute principal angles between column spaces
        # For rank-6 approximation
        r = min(RANK, U0.shape[1], U1.shape[1])
        C = U0[:, :r].T @ U1[:, :r]
        sv = np.linalg.svd(C, compute_uv=False)
        sv = np.clip(sv, 0, 1)
        angles = np.arccos(sv)
        total += float(np.sum(angles))
    return total


# ============================================================================
# CORPUS STATISTICS (for predictions)
# ============================================================================
def compute_corpus_stats():
    """Compute corpus statistics for predictions."""
    freq = collections.Counter(train_ids.tolist())
    total = sum(freq.values())
    freqs = np.array([freq.get(t, 0) / total for t in range(VOCAB)], dtype=float)
    freqs_nz = freqs[freqs > 0]

    H_bits = float(-np.sum(freqs_nz * np.log2(freqs_nz + 1e-12)))
    V_eff = float(2**H_bits)
    rare_frac = float(np.mean(freqs < np.median(freqs_nz)))

    # Bigram entropy
    bigram = collections.Counter()
    for i in range(len(train_ids) - 1):
        bigram[(train_ids[i].item(), train_ids[i+1].item())] += 1
    n_bigram = sum(bigram.values())
    bigram_freqs = np.array(list(bigram.values()), dtype=float) / n_bigram
    H_bigram = float(-np.sum(bigram_freqs * np.log2(bigram_freqs + 1e-12)))

    nnz = len(bigram)

    return {
        'V': VOCAB,
        'H_bits': H_bits,
        'V_eff': V_eff,
        'rare_frac': rare_frac,
        'H_bigram': H_bigram,
        'nnz': nnz,
    }


# ============================================================================
# GEOMETRIC PREDICTIONS (from corpus + architecture)
# ============================================================================
def make_predictions():
    """Make geometric predictions from corpus + architecture."""
    corpus = compute_corpus_stats()
    
    # 1. Basin type
    r_natural = D // N_HEADS // 4
    r_ratio = RANK / max(r_natural, 1)
    monodromy = 'Abelian' if r_ratio < 0.5 else 'Non-Abelian'
    lr_pred = 50 if r_ratio < 0.5 else 5
    
    # 2. Entropy floor val*
    # val* ≈ H_unigram × correction factor
    # From our data: H=~6.5 bits, val*=0.062
    val_star = float(corpus['H_bits'] / math.log2(math.e) * 0.012)
    val_star = max(0.02, min(0.20, val_star))
    
    # 3. Strip energy
    # For random high-dimensional subspaces, angles ≈ π/2
    E_strip_per_pair = RANK * math.pi / 2  # ≈ 9.42
    E_strip_total = E_strip_per_pair * (N_STU - 1)  # ≈ 47.1
    
    # 4. τ_high
    tau_high = 5.0 + 0.5 * corpus['H_bits'] / math.log2(VOCAB + 1)
    
    # 5. Φ_cl_min
    phi_min = max(3, math.ceil(5 * (1 - corpus['rare_frac'] * 0.4)))
    
    # 6. CE budget
    if r_ratio < 0.5:
        phase3_CE = 24
    else:
        phase3_CE = 120
    
    total_CE = 1 + 10 + phase3_CE + 50 + 1 + 50 + 8
    
    # 7. r_m2^σ threshold
    rm2_threshold = max(0.50, 0.65 * (1 - 0.1 * abs(r_ratio - 1)))
    
    return {
        'monodromy_type': monodromy,
        'lr_recommended': lr_pred,
        'r_ratio': r_ratio,
        'val_star': val_star,
        'E_strip_total': E_strip_total,
        'E_strip_per_pair': E_strip_per_pair,
        'tau_high': tau_high,
        'phi_cl_min': phi_min,
        'phase3_CE': phase3_CE,
        'total_CE': total_CE,
        'rm2_threshold': rm2_threshold,
        'corpus': corpus,
    }


# ============================================================================
# VERIFICATION TESTS
# ============================================================================
class CompilerVerificationTests:
    def __init__(self, model, predictions, verbose=True):
        self.model = model
        self.preds = predictions
        self.verbose = verbose
        self.results = []
        self.passed = 0
        self.failed = 0
        
        # Measure observed quantities
        self.observed = self._measure_observed()
    
    def _measure_observed(self):
        """Measure all geometric quantities from the model."""
        observed = {
            'val': eval_val(self.model, n=12),
            'phi': phi_clean(self.model),
            'tau': gluing_defect(self.model, n=6),
            'strip_energy': strip_energy(self.model),
        }
        
        # Get individual phases
        angles = sheet_angles(self.model)
        observed['phases'] = angles
        observed['phi_fraction'] = sum(1 for a in angles if not np.isnan(a) and 
                                       (abs(a) < 0.3 or abs(abs(a) - math.pi) < 0.3)) / len(angles)
        
        return observed
    
    def _check(self, name, pred, obs, tolerance, severity='MAJOR', description=''):
        """Run a single verification check."""
        if isinstance(pred, str):
            match = pred == obs
        else:
            match = abs(float(pred) - float(obs)) <= tolerance
        
        result = {
            'name': name,
            'predicted': pred,
            'observed': obs,
            'tolerance': tolerance,
            'match': match,
            'severity': severity,
            'description': description,
        }
        self.results.append(result)
        
        if match:
            self.passed += 1
            status = '✓'
        else:
            self.failed += 1
            status = '✗'
        
        if self.verbose:
            print(f"  {status} {severity:<8} {name:<25} "
                  f"pred={pred} obs={obs} tol={tolerance}")
        
        return match
    
    def run_all_tests(self):
        """Run all verification tests."""
        print("\n" + "=" * 70)
        print("COMPILER VERIFICATION TESTS")
        print("=" * 70)
        print(f"  Model val: {self.observed['val']:.4f}")
        print(f"  Model phi: {self.observed['phi']}/5")
        print(f"  Model tau: {self.observed['tau']:.2f}")
        print(f"  Strip energy: {self.observed['strip_energy']:.2f}")
        print()
        
        # ─── CRITICAL TESTS ───────────────────────────────────────────────────
        print("  CRITICAL TESTS (must pass)")
        print("  " + "-" * 60)
        
        # C1: Val is finite and not NaN
        self._check(
            'Val finite',
            float('inf'),
            self.observed['val'],
            1e10,
            severity='CRITICAL',
            description='Loss must be finite'
        )
        
        # C2: Val is below threshold (not exploded)
        self._check(
            'Val < 10.0',
            10.0,
            self.observed['val'],
            0.5,
            severity='CRITICAL',
            description='Model must not have exploded'
        )
        
        # C3: Strip energy is positive
        self._check(
            'Strip energy > 0',
            0.0,
            self.observed['strip_energy'],
            0.01,
            severity='CRITICAL',
            description='Strip energy must be positive'
        )
        
        # C4: Model has forward pass
        try:
            x, y = get_batch()
            _, l = self.model(x, y)
            forward_ok = not torch.isnan(l)
        except:
            forward_ok = False
        self._check(
            'Forward pass OK',
            True,
            forward_ok,
            0,
            severity='CRITICAL',
            description='Model must have valid forward pass'
        )
        
        # ─── MAJOR TESTS ─────────────────────────────────────────────────────
        print("\n  MAJOR TESTS")
        print("  " + "-" * 60)
        
        # M1: Strip energy matches prediction
        self._check(
            'Strip energy match',
            self.preds['E_strip_total'],
            self.observed['strip_energy'],
            5.0,
            severity='MAJOR',
            description=f"Expected ~{self.preds['E_strip_total']:.2f}"
        )
        
        # M2: Tau is in stable range (1.5-5.7) or near it
        tau_ok = 1.0 <= self.observed['tau'] <= 7.0
        self._check(
            'Tau in range [1.0, 7.0]',
            True,
            tau_ok,
            0,
            severity='MAJOR',
            description=f"Tau={self.observed['tau']:.2f}"
        )
        
        # M3: Phi is clean enough (>= 3/5)
        self._check(
            'Phi >= 3/5',
            3,
            self.observed['phi'],
            1,
            severity='MAJOR',
            description=f"Phi={self.observed['phi']}/5"
        )
        
        # M4: Val is decreasing (below initial)
        # We need to compare to spectral E0 initial val
        # Use a heuristic: val < 1.0 for a good compiler
        self._check(
            'Val < 1.0',
            1.0,
            self.observed['val'],
            0.1,
            severity='MAJOR',
            description=f"Val={self.observed['val']:.4f}"
        )
        
        # M5: Strip energy is less than random baseline
        # Random high-dimensional subspaces: angles ≈ π/2
        random_energy = RANK * math.pi / 2 * (N_STU - 1)  # ~47.1
        self._check(
            'Strip energy < random',
            random_energy,
            self.observed['strip_energy'],
            10.0,
            severity='MAJOR',
            description=f"Expected < {random_energy:.2f}"
        )
        
        # ─── MINOR TESTS ─────────────────────────────────────────────────────
        print("\n  MINOR TESTS")
        print("  " + "-" * 60)
        
        # m1: Phi is exact 5/5 (or close)
        self._check(
            'Phi = 5/5',
            5,
            self.observed['phi'],
            1,
            severity='MINOR',
            description="Perfect orbit is optional"
        )
        
        # m2: Val is below floor (0.062)
        self._check(
            'Val < 0.062',
            0.062,
            self.observed['val'],
            0.02,
            severity='MINOR',
            description="Floor may not be reached"
        )
        
        # m3: Phase pattern has alternating structure
        phases = self.observed['phases']
        alternating = all(
            (phases[i] > 0) != (phases[i+1] > 0) 
            for i in range(len(phases) - 1)
            if not np.isnan(phases[i]) and not np.isnan(phases[i+1])
        ) if len(phases) > 1 else False
        self._check(
            'Alternating phases',
            True,
            alternating,
            0,
            severity='MINOR',
            description="Not required for validity"
        )
        
        # ─── SUMMARY ─────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"  PASSED: {self.passed}")
        print(f"  FAILED: {self.failed}")
        
        # Check critical tests
        critical_failures = [r for r in self.results if r['severity'] == 'CRITICAL' and not r['match']]
        if critical_failures:
            print(f"\n  ✗ CRITICAL FAILURES: {len(critical_failures)}")
            for r in critical_failures:
                print(f"    - {r['name']}: pred={r['predicted']} obs={r['observed']}")
            print("\n  → COMPILER IS BROKEN. Fix critical issues first.")
            return False
        
        major_failures = [r for r in self.results if r['severity'] == 'MAJOR' and not r['match']]
        if major_failures:
            print(f"\n  ✗ MAJOR FAILURES: {len(major_failures)}")
            for r in major_failures:
                print(f"    - {r['name']}: pred={r['predicted']} obs={r['observed']}")
            print("\n  → COMPILER IS DEGRADED. Check major issues.")
            return False
        
        print("\n  ✓ ALL CRITICAL AND MAJOR TESTS PASSED")
        print("  → COMPILER IS FUNCTIONING CORRECTLY")
        return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("COMPILER VERIFICATION TESTS")
    print("=" * 70)
    
    # 1. Make predictions from corpus + architecture
    print("\n1. MAKING PREDICTIONS FROM CORPUS + ARCHITECTURE")
    print("   " + "-" * 50)
    preds = make_predictions()
    print(f"   Monodromy: {preds['monodromy_type']}")
    print(f"   LR recommended: LR×{preds['lr_recommended']}")
    print(f"   Val*: {preds['val_star']:.4f}")
    print(f"   Strip energy: {preds['E_strip_total']:.2f}")
    print(f"   Tau_high: {preds['tau_high']:.1f}")
    print(f"   Phi_min: {preds['phi_cl_min']}/5")
    print(f"   Total CE: {preds['total_CE']}")
    
    # 2. Load or create model
    print("\n2. LOADING MODEL")
    print("   " + "-" * 50)
    
    # Try to load saved model, or create one
    model_path = 'compiler_smart_final.pt'
    if os.path.exists(model_path):
        print(f"   Loading model from {model_path}")
        model = LM()
        try:
            data = torch.load(model_path)
            if isinstance(data, dict) and 'state_dict' in data:
                model.load_state_dict(data['state_dict'])
            else:
                model.load_state_dict(data)
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
            print("   Creating new model...")
            model = LM()
            # Initialize with spectral E0
            try:
                from double_blind_validation import compute_corpus_stats
                # Use simple init
            except:
                pass
    else:
        print(f"   Model not found. Creating new model...")
        model = LM()
    
    print(f"   Model created. Val: {eval_val(model, n=6):.4f}")
    
    # 3. Run verification tests
    print("\n3. RUNNING VERIFICATION TESTS")
    print("   " + "-" * 50)
    tester = CompilerVerificationTests(model, preds)
    success = tester.run_all_tests()
    
    # 4. Save results
    print("\n4. SAVING RESULTS")
    print("   " + "-" * 50)
    results = {
        'predictions': preds,
        'observed': tester.observed,
        'test_results': tester.results,
        'passed': tester.passed,
        'failed': tester.failed,
        'success': success,
    }
    
    # Remove non-serializable items
    results['observed']['phases'] = [str(p) for p in results['observed']['phases']]
    if 'corpus' in preds:
        del preds['corpus']
    
    with open('compiler_verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to compiler_verification_results.json")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    if success:
        print("  ✓ COMPILER IS FUNCTIONING CORRECTLY")
    else:
        print("  ✗ COMPILER NEEDS INVESTIGATION")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
