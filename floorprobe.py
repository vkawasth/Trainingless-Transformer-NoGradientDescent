"""IS THE CAPACITY FILTER VISIBLE IN THE FLOOR ITSELF?

Phase 16 found a reproducible capacity boundary:

    D      lambda_min s17   s23     final val   phi > 90 deg
    48        +0.054      +0.049      1.74        no
    64        +0.040      +0.032      1.24        no
    96        +0.056      +0.055      0.78        no
    128       -0.308      -0.068      0.41        BOTH SEEDS
    176       -1.712      +0.011      0.23        one seed

Below 128 the curvature stays positive in every run; at 128 both seeds cross
into negative curvature, and the loss knee is at the same place (0.78 -> 0.41).
Two seeds is the artefact control -- a numerical excursion would not reproduce.

The question this asks: does the filter show up in L_floor, which the compiler
measures BEFORE Phase 1 from 200 AdamW steps? If depth and width gate which
corpus relations can be expressed, a model below threshold cannot reach the same
floor, and L_floor should drop sharply across the boundary and then flatten.

If so, L_floor / ln(VOCAB) -- the floor as a fraction of the uniform-distribution
loss, measured at 0.104 for D=256 -- is a JOINT corpus-architecture invariant
rather than a corpus one, which is the map we are looking for.

Cheap: only the floor probe runs, not the pipeline. VOCAB is fixed by the corpus
so ln(V) is common to every row, and the comparison is clean.
"""
import json, subprocess, io, contextlib, math, re, numpy as np

subprocess.run(["python3", "/mnt/user-data/uploads/build_corpus.py", "--out", "/tmp",
                "--loops", "300"], check=True, capture_output=True)
RAW = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT = RAW.index("# \u2500\u2500 INIT MODEL")
HEAD = RAW[:CUT]

rows = []
print("  floor probe only -- 200 AdamW steps, no pipeline\n")
print(f"  {'D':>5}{'heads':>7}{'P':>12}{'L_floor':>10}{'|g_floor|':>11}"
      f"{'L/lnV':>9}{'|g|^2/L':>10}")
for D, NH in ((48, 4), (64, 4), (96, 4), (128, 4), (176, 4), (256, 4)):
    src = HEAD.replace("D=256; N_HEADS=4", f"D={D}; N_HEADS={NH}", 1)
    G = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(src, G)
    except Exception as e:
        print(f"  {D:>5}  FAILED {type(e).__name__}: {str(e)[:50]}")
        continue
    out = buf.getvalue()
    m = re.search(r"val=([\d.]+)\s+\|\|g_floor\|\|=([\d.]+)", out)
    if not m:
        print(f"  {D:>5}  no floor line")
        continue
    Lf, gf = float(m.group(1)), float(m.group(2))
    V = G["VOCAB"]
    P = sum(p.numel() for p in G["m_floor"].parameters())
    r = dict(D=D, NH=NH, P=P, Lf=Lf, gf=gf, V=V,
             LlnV=Lf / math.log(V), rate=gf ** 2 / Lf)
    rows.append(r)
    print(f"  {D:>5}{NH:>7}{P:>12,}{Lf:>10.4f}{gf:>11.4f}"
          f"{r['LlnV']:>9.4f}{r['rate']:>10.4f}", flush=True)
    del G
    import gc; gc.collect()

json.dump(rows, open("/home/claude/work/res_floor.json", "w"), indent=2)
if len(rows) >= 3:
    D = np.array([r["D"] for r in rows], float)
    Lf = np.array([r["Lf"] for r in rows])
    rt = np.array([r["rate"] for r in rows])
    print(f"\n  L_floor drop between consecutive capacities:")
    for i in range(1, len(rows)):
        print(f"    D {rows[i-1]['D']:>3} -> {rows[i]['D']:>3}   "
              f"{Lf[i-1]:.4f} -> {Lf[i]:.4f}   "
              f"{100*(Lf[i]-Lf[i-1])/Lf[i-1]:+7.1f}%")
    print(f"\n  a sharp drop then a flattening => the filter is in the floor, and")
    print(f"  L_floor/ln(V) is a joint corpus-architecture invariant")
    print(f"  smooth monotone decline    => capacity scales the floor, no filter")
    print(f"\n  |g_floor|^2 / L_floor by D: " +
          "  ".join(f"{r['D']}:{r['rate']:.4f}" for r in rows))
    print(f"  (measured k for D=256 was 0.0434; ratio k/(|g|^2/L) was 0.571)")
