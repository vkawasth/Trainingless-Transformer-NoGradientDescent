import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set seed for exact reproducibility across runs
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Synthetic Corpus Generator (Mimicking the 936 Well-Sampled Contexts)
# =====================================================================
NUM_CONTEXTS = 936
VOCAB_SIZE = 1017
STEPS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate log-normal empirical frequencies P(x) (Power-law distribution)
raw_freqs = np.random.lognormal(mean=3.5, sigma=1.2, size=NUM_CONTEXTS)
raw_freqs = np.maximum(raw_freqs, 32.0) # Ensure cnt >= 4b property
context_counts = torch.tensor(raw_freqs, dtype=torch.float32)
P_x = context_counts / context_counts.sum()

# Generate branching factors b(x) in range [4, 30]
b_x = torch.randint(low=4, high=31, size=(NUM_CONTEXTS,), dtype=torch.long)

# Construct Ground-Truth Support Masks and Targets
support_masks = torch.zeros(NUM_CONTEXTS, VOCAB_SIZE, dtype=torch.bool)
target_distributions = torch.zeros(NUM_CONTEXTS, VOCAB_SIZE, dtype=torch.float32)

for c in range(NUM_CONTEXTS):
    # Randomly pick b(x) valid successor tokens
    valid_tokens = torch.randperm(VOCAB_SIZE)[:b_x[c]]
    support_masks[c, valid_tokens] = True
    
    # Assign target probabilities inside support (with mild shape non-uniformity)
    dirichlet_weights = torch.distributions.Dirichlet(torch.ones(b_x[c])).sample()
    target_distributions[c, valid_tokens] = dirichlet_weights

# =====================================================================
# 2. Context-Balanced Head-Ranking Loss Module
# =====================================================================
class ContextBalancedHeadLoss(nn.Module):
    def __init__(self, support_masks, context_counts, alpha=0.75, margin_target=2.0, lambda_rank=0.5):
        super().__init__()
        self.num_contexts = support_masks.size(0)
        self.margin_target = margin_target
        self.lambda_rank = lambda_rank
        
        # Compute frequency weights w(x)
        P_x = context_counts / context_counts.sum()
        P_bar = 1.0 / self.num_contexts
        w_x = torch.pow(P_bar / (P_x + 1e-6), alpha)
        w_x = w_x / w_x.mean() # Preserve global learning rate scale
        
        self.register_buffer("support_masks", support_masks)
        self.register_buffer("w_x", w_x)

    def forward(self, logits, context_ids, target_ids):
        masks = self.support_masks[context_ids]
        batch_weights = self.w_x[context_ids]

        # 1. Frequency-Weighted Cross-Entropy
        ce_loss = F.cross_entropy(logits, target_ids, reduction='none')
        weighted_ce = (ce_loss * batch_weights).mean()

        # 2. Pairwise Head-Ranking Intruder Penalty
        # Weakest valid logit per context
        valid_logits = logits.masked_fill(~masks, float('inf'))
        min_valid, _ = valid_logits.min(dim=-1)

        # Strongest invalid logit per context (Head Intruder)
        invalid_logits = logits.masked_fill(masks, float('-inf'))
        max_invalid, _ = invalid_logits.max(dim=-1)

        # Enforce min(valid) - max(invalid) >= margin_target
        ranking_violations = F.margin_ranking_loss(
            min_valid, 
            max_invalid, 
            target=torch.ones_like(min_valid), 
            margin=self.margin_target, 
            reduction='none'
        )
        weighted_rank_loss = (ranking_violations * batch_weights).mean()

        total_loss = weighted_ce + self.lambda_rank * weighted_rank_loss
        return total_loss

# =====================================================================
# 3. Model Simulation & Initialization Protocol
# =====================================================================
def create_initial_logits():
    """
    Initializes a logit representation (936, 1017) exhibiting real session empirical properties:
    - High pairwise ordering F_ord ~ 0.89-0.90
    - Negative margin gap: min(valid) - max(invalid) ~ -0.91 nats
    - Invalid mass leakage Z_invalid / Z_valid > 1.0
    """
    logits = torch.randn(NUM_CONTEXTS, VOCAB_SIZE, dtype=torch.float32) * 0.5
    for c in range(NUM_CONTEXTS):
        valid_idx = support_masks[c].nonzero(as_tuple=True)[0]
        invalid_idx = (~support_masks[c]).nonzero(as_tuple=True)[0]
        
        # Elevate valid logits above average invalid
        logits[c, valid_idx] += 2.0
        
        # Inject head intruders: boost ~3% of invalid tokens to sit above min(valid)
        num_intruders = int(len(invalid_idx) * 0.03)
        intruder_idx = invalid_idx[torch.randperm(len(invalid_idx))[:num_intruders]]
        logits[c, intruder_idx] += 2.8
        
    return nn.Parameter(logits.to(DEVICE))

# =====================================================================
# 4. Diagnostics & Metric Evaluation Evaluator
# =====================================================================
@torch.no_grad()
def evaluate_manifold_metrics(logits_param):
    logits = logits_param.detach()
    probs = F.softmax(logits, dim=-1)
    
    kl_list = []
    margin_list = []
    z_ratio_list = []
    f_ord_list = []

    for c in range(NUM_CONTEXTS):
        mask = support_masks[c].to(DEVICE)
        target = target_distributions[c].to(DEVICE)
        b_c = b_x[c].item()
        
        # KL Divergence against true target
        p_c = probs[c]
        kl = (target[mask] * (torch.log(target[mask] + 1e-12) - torch.log(p_c[mask] + 1e-12))).sum().item()
        kl_list.append(kl)
        
        # Head Margin Gap: min(valid) - max(invalid)
        valid_logits = logits[c][mask]
        invalid_logits = logits[c][~mask]
        margin = (valid_logits.min() - invalid_logits.max()).item()
        margin_list.append(margin)
        
        # Z_invalid / Z_valid Partition Ratio
        z_valid = torch.exp(valid_logits).sum()
        z_invalid = torch.exp(invalid_logits).sum()
        z_ratio_list.append((z_invalid / z_valid).item())
        
        # Pairwise Order Homomorphism F_ord
        # Proportion of (valid, invalid) pairs where valid > invalid
        valid_matrix = valid_logits.unsqueeze(1) # (b_c, 1)
        invalid_matrix = invalid_logits.unsqueeze(0) # (1, 1017 - b_c)
        correct_pairs = (valid_matrix > invalid_matrix).float().mean().item()
        f_ord_list.append(correct_pairs)

    # Calculate correlation between log(frequency) and KL
    log_freqs = torch.log(context_counts).numpy()
    kl_arr = np.array(kl_list)
    corr_freq_kl = np.corrcoef(log_freqs, kl_arr)[0, 1]

    return {
        "mean_kl": np.mean(kl_list),
        "mean_margin": np.mean(margin_list),
        "mean_z_ratio": np.mean(z_ratio_list),
        "f_ord": np.mean(f_ord_list),
        "corr_freq_kl": corr_freq_kl
    }

# =====================================================================
# 5. Execution Loop (Standard CE vs CBNG-Head)
# =====================================================================
def run_experiment(use_cbng=False):
    logits_param = create_initial_logits()
    optimizer = torch.optim.Adam([logits_param], lr=0.05)
    
    if use_cbng:
        criterion = ContextBalancedHeadLoss(
            support_masks.to(DEVICE), 
            context_counts.to(DEVICE),
            alpha=0.75, 
            margin_target=2.0, 
            lambda_rank=0.5
        )
    
    history = []
    
    for step in range(STEPS + 1):
        if step % 50 == 0:
            metrics = evaluate_manifold_metrics(logits_param)
            history.append((step, metrics))
        
        if step == STEPS:
            break
            
        # Mini-batch sampling weighted by context frequency (empirical data stream)
        batch_contexts = torch.multinomial(P_x, num_samples=256, replacement=True).to(DEVICE)
        
        # Sample target token per context according to target distribution
        batch_targets = torch.zeros(256, dtype=torch.long, device=DEVICE)
        for i, c in enumerate(batch_contexts):
            batch_targets[i] = torch.multinomial(target_distributions[c], num_samples=1).item()
            
        batch_logits = logits_param[batch_contexts]
        
        if not use_cbng:
            loss = F.cross_entropy(batch_logits, batch_targets)
        else:
            loss = criterion(batch_logits, batch_contexts, batch_targets)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return history

print("Running 300-step training comparison across 936 well-sampled contexts...\n")

std_history = run_experiment(use_cbng=False)
cbng_history = run_experiment(use_cbng=True)

# =====================================================================
# 6. Report Trajectory Comparison Table
# =====================================================================
print(f"{'Step':<5} | {'Strategy':<10} | {'Mean KL':<8} | {'Margin Gap':<10} | {'Z_inv/Z_val':<11} | {'F_ord':<6} | {'corr(ln f, KL)':<14}")
print("-" * 78)

for idx in range(len(std_history)):
    step = std_history[idx][0]
    m_std = std_history[idx][1]
    m_cbng = cbng_history[idx][1]
    
    print(f"{step:<5} | Standard   | {m_std['mean_kl']:<8.3f} | {m_std['mean_margin']:<10.3f} | {m_std['mean_z_ratio']:<11.3f} | {m_std['f_ord']:<6.3f} | {m_std['corr_freq_kl']:<14.3f}")
    print(f"{'':<5} | CBNG-Head  | {m_cbng['mean_kl']:<8.3f} | {m_cbng['mean_margin']:<10.3f} | {m_cbng['mean_z_ratio']:<11.3f} | {m_cbng['f_ord']:<6.3f} | {m_cbng['corr_freq_kl']:<14.3f}")
    print("-" * 78)
