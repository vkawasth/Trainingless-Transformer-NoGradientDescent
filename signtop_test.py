from signtop import signtop_compress

u = {name: theta_after[name] - theta_before[name] for name in mats}
u_hat = signtop_compress(u, rank_budget=32)
