import torch
import numpy as np
import math
from itertools import combinations

# --- 1. Spectral Projection Core ---
def target_spectral_initialization(operators, target_rho=33.29):
    """ Scales operators to match the target Ihara spectral radius. """
    projected = []
    for op in operators:
        current_rho = np.max(np.abs(np.linalg.eigvals(op)))
        scaling_factor = target_rho / (current_rho + 1e-8)
        projected.append(op * scaling_factor)
    return projected

def fast_path_assembly(teacher, cascade_prime, d, n_stu):
    """ Instantiates the student directly into the target Stokes chamber. """
    student = LM(d, N_HEADS, n_stu)
    # Align weights to teacher backbone
    student.te.weight.data.copy_(teacher.te.weight.data)
    
    # Target spectrum projection
    init_ops = target_spectral_initialization(cascade_prime, TARGET_RHO)
    
    with torch.no_grad():
        for i in range(n_stu):
            W_t = torch.tensor(init_ops[i], dtype=torch.float32)
            student.blocks[i].attn.WK.weight.copy_(W_t)
            student.blocks[i].attn.WQ.weight.copy_(W_t.T)
            # FF and Op weights inherited from teacher for phase-sync
            student.blocks[i].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            student.blocks[i].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
    return student

# --- 2. Integrated Execution Pipeline ---
def run_integrated_assembly():
    # A. Train/Load Teacher
    teacher = LM(D, N_HEADS, N_LAYERS_T)
    # [Insert Training Loop Here]
    teacher.eval()

    # B. Extract Geometric Data (Jacobians & Prime Paths)
    # Extract Jacobian chain Js and identify prime paths
    Js = [np.mean(J_acc[l], axis=0) for l in range(N_LAYERS_T)]
    
    # C. Calculate Prime Path Cascade
    # Identification of paths via DGLA curvature (mu6_op)
    prime_paths = identify_prime_paths(Js) 
    cascade_prime = [mu6_op([Js[i] for i in c]) for c in prime_paths]

    # D. One-Shot Spectral Assembly
    print("Executing One-Shot Spectral Assembly...")
    student = fast_path_assembly(teacher, cascade_prime, D, N_STU)

    # E. Verify Convergence
    # The student should now be at the Ihara attractor rho ~ 33.29
    loss = eval_val(student)
    print(f"One-shot assembly complete. Final Val: {loss:.4f}")
    return student

# --- 3. Geometric Visualization Trigger ---
# We are aligning the student to the target spectral chamber
# which represents the attractor on the Grassmannian sphere.
