"""Stress test for structured posteriors: a deliberately bimodal posterior mu = 1/2 mu_A + 1/2 mu_B whose components have
different latent / factor / subspace structure. Does multiplication keep A != B, or flatten them into their mixture?

A: chain  z -> x1 -> x2   (latent z; factors {p(z), p(x1|z), p(x2|x1)}; structural subspace span(e1))
B: common cause  x1 <- w -> x2   (latent w; factors {p(w), p(x1|w), p(x2|w)}; subspace span(e2))
The mixture is formed by monad multiplication over a hypothesis H ~ (1/2, 1/2). C is a single unstructured posterior with
EXACTLY the same marginal on (x1, x2) and record {saturated table}; D is a single posterior whose record is the union of
A's and B's structure. Then evidence e = "x2 = 1" (likelihood 1[x2 = 1]) is conditioned on. Everything exact."""
import os, sys, json
from fractions import Fraction as Fr
from itertools import product
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from amb_vigneaux import sp

M = sp.Product(sp.FactorSet, sp.Trace, sp.Subspace(2)); E1, E2 = [1, 0], [0, 1]

def marg(joint, keep):
    out = {}
    for k, p in joint.items(): kk = tuple(k[i] for i in keep); out[kk] = out.get(kk, 0) + p
    return out

def mu_A():                                   # chain over latent z
    j = {}
    for z, x1, x2 in product((0, 1), repeat=3):
        pz = Fr(1, 2); px1 = Fr(4, 5) if x1 == z else Fr(1, 5); px2 = Fr(9, 10) if x2 == x1 else Fr(1, 10)
        j[(z, x1, x2)] = pz * px1 * px2
    return marg(j, (1, 2))

def mu_B():                                   # common cause w
    j = {}
    for w, x1, x2 in product((0, 1), repeat=3):
        pw = Fr(3, 10) if w else Fr(7, 10); px1 = Fr(3, 4) if x1 == w else Fr(1, 4); px2 = Fr(19, 20) if x2 == w else Fr(1, 20)
        j[(w, x1, x2)] = pw * px1 * px2
    return marg(j, (1, 2))

REC = {"A": (sp.FactorSet.of("p(z)", "p(x1|z)", "p(x2|x1)"), sp.Trace.unit, M.parts[2].of(E1)),
       "B": (sp.FactorSet.of("p(w)", "p(x1|w)", "p(x2|w)"), sp.Trace.unit, M.parts[2].of(E2))}
MU = {"A": mu_A(), "B": mu_B()}

if __name__ == "__main__":
    PB, GL = sp.PerBranch(M), sp.Global(M)
    prior_H = {("A", M.unit): Fr(1, 2), ("B", M.unit): Fr(1, 2)}
    # monad multiplication: H -> structured posterior of that hypothesis
    rho_pb = PB.bind(prior_H, lambda h: {(x, REC[h]): p for x, p in MU[h].items()})
    rho_gl = GL.bind(({"A": Fr(1, 2), "B": Fr(1, 2)}, M.unit), lambda h: (MU[h], REC[h]))
    mix = PB.U(rho_pb)
    rho_C_pb = {(x, (sp.FactorSet.of("saturated table"), (), M.parts[2].unit)): p for x, p in mix.items()}
    rho_D_gl = (mix, (REC["A"][0] | REC["B"][0], (), M.parts[2].mul(REC["A"][2], REC["B"][2])))
    out = {}
    out["giry_mixture_equals_C"] = PB.U(rho_pb) == PB.U(rho_C_pb) == GL.U(rho_gl)
    out["global_equals_D_(A_and_B_vs_A_or_B)"] = rho_gl == rho_D_gl
    out["per_branch_records"] = len(PB.records(rho_pb))
    out["per_branch_recovers_A_B_exactly"] = all(PB.given_record(rho_pb, REC[h]) == MU[h] for h in "AB")
    out["MI_outcome_record_nats"] = sp.mutual_information({(x, m): p for (x, m), p in rho_pb.items()})
    # evidence: x2 = 1
    L = lambda x: Fr(1) if x[1] == 1 else Fr(0)
    post_pb = PB.condition(rho_pb, L, e="x2=1", factor="1[x2=1]", trace_slot=1, factor_slot=0)
    post_gl = GL.condition(rho_gl, L, e="x2=1", factor="1[x2=1]", trace_slot=1, factor_slot=0)
    rec_post = PB.records(post_pb)
    stamp = lambda h: (REC[h][0] | {"1[x2=1]"}, ("x2=1",), REC[h][2])
    pA = rec_post[stamp("A")]; pB = rec_post[stamp("B")]
    zA = sum(p for x, p in MU["A"].items() if x[1] == 1); zB = sum(p for x, p in MU["B"].items() if x[1] == 1)
    out["P(A|e)_per_branch"] = str(pA); out["P(A|e)_bayes"] = str(zA / (zA + zB)); out["bayes_factor_A:B"] = str(zA / zB)
    out["per_branch_matches_bayes"] = pA == zA / (zA + zB)
    out["global_after_e_record"] = sorted(post_gl[1][0]); out["global_after_e_can_weigh_A_vs_B"] = False
    out["giry_posterior_x1_given_e"] = {str(k): str(v) for k, v in PB.U(post_pb).items()}
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1, default=str)
