python3 patch_zoneadam.py --in compiler_geometri_patched_86.py --out mybuild.py \
  --no-zone1 --eta-mf 0.002 --p3cap 300 --p3min 300 --p3plat 0.0005 \
  --p3valstop 0 --no-geostop --no-snapper --vfreeze 0 \
  --prune-attn --struct-loss --no-topogate



#Drop --struct-loss for the control — that's the comparison that shows the 0.32-nat gain.
#Four lines confirm the flags took:
#[zoneadam] STRUCTURAL LOSS: LSE boundary on observed support, gamma=10.0 lambda=0.5 fork=140
#  [zoneadam] PRUNED attention QK in blocks [0] (131,072 params frozen; floor p_max=0.0741)
#  [zoneadam] SNAPPER SKIPPED -- no convex basin on this corpus
#  [zoneadam] TOPOGATE REVERTED
