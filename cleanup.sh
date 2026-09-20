#!/usr/bin/env bash
# cleanup.sh -- verify the keepers reproduce, THEN delete the dead ends.
#
#   bash cleanup.sh          dry run: rebuild, verify, list what would go
#   bash cleanup.sh --delete same, but actually delete (only if verify passed)
#
# Keeps: p1.pt m1.pt m2.pt m3.pt rhm10.py build_corpus_rhm4.py
# Those five files reproduce every result in the project with no retraining.

set -u
DELETE=0; [ "${1:-}" = "--delete" ] && DELETE=1

KEEP=(p1.pt m1.pt m2.pt m3.pt rhm10.py build_corpus_rhm4.py)
JUNK=(q1.pt r1.pt
      build_corpus_prod.py sanity_product.py
      build_corpus_rhm.py build_corpus_rhm2.py build_corpus_rhm3.py
      rhm2.py rhm3.py rhm4.py rhm5.py rhm6.py rhm7.py rhm8.py rhm9.py
      c_probe.py c_probe2.py c_probe3.py c_probe4.py c_probe5.py
      lr_probe.py)

echo "== 1. keepers present?"
missing=0
for f in "${KEEP[@]}"; do
  if [ -f "$f" ]; then echo "   ok      $f"
  else echo "   MISSING $f"; missing=1; fi
done
[ $missing -eq 1 ] && { echo "!! a keeper is missing -- deleting nothing"; exit 1; }

echo
echo "== 2. rebuild the corpus p1.pt was trained on"
python3 build_corpus_rhm4.py --out /tmp --poset --reverse-valid \
        --nrules 12 --poset-density 0.25 --nleaf 48 --n-train 200000 \
        > /tmp/_build.log 2>&1 || { echo "!! build failed, see /tmp/_build.log"; exit 1; }
grep -E "C_trans |C_incomp |PARTIAL ORDER" /tmp/_build.log | sed 's/^/   /'

echo
echo "== 3. reproduce the headline numbers from p1.pt"
python3 rhm10.py --mlp --load p1.pt > /tmp/_verify.log 2>&1 \
        || { echo "!! probe run failed, see /tmp/_verify.log"; exit 1; }
grep -E "D_ord=|within/between|D_trans=" /tmp/_verify.log | sed 's/^/   /'

# the three anchors, as measured. If these drift, the corpus does not match
# the checkpoint and nothing should be deleted.
python3 - <<'PY'
import re, sys
t = open('/tmp/_verify.log').read()
want = [("D_ord h2",      r"h2: D_ord=([\d.]+)",                 0.4551),
        ("within/between", r"ratio within/between\s+=\s+([\d.]+)", 0.3677),
        ("D_trans h2",    r"h2:  D_trans=([\d.]+)",              0.5075)]
bad = 0
print()
for name, pat, exp in want:
    m = re.search(pat, t)
    if not m:
        print(f"   {name:<16} NOT FOUND"); bad = 1; continue
    got = float(m.group(1)); ok = abs(got - exp) < 0.01
    print(f"   {name:<16} got {got:.4f}  expected {exp:.4f}  {'ok' if ok else 'MISMATCH'}")
    bad |= (not ok)
sys.exit(bad)
PY
[ $? -ne 0 ] && { echo; echo "!! numbers do not reproduce -- deleting nothing"; exit 1; }

echo
echo "== 4. dead ends"
total=0
for f in "${JUNK[@]}"; do
  [ -f "$f" ] || continue
  sz=$(du -k "$f" | cut -f1); total=$((total+sz))
  if [ $DELETE -eq 1 ]; then rm -f "$f"; echo "   deleted  $f (${sz}K)"
  else echo "   would rm $f (${sz}K)"; fi
done
echo "   ---- ${total}K total"

echo
if [ $DELETE -eq 1 ]; then
  echo "done. /tmp/{train,val}_ids.json are regenerable -- clear them too if needed."
else
  echo "dry run. rerun with:  bash cleanup.sh --delete"
fi
