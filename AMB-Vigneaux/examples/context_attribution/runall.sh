#!/bin/bash
# full study: document contexts (ARCH=docs) at n = 50, 100, 200 tokens, and the absorbed-reference contrast (ARCH=ctx)
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
(NTOK=50 REPS=40 ARCH=docs python run_docs.py > full_docs_n50.txt 2>&1; NTOK=200 REPS=40 ARCH=docs python run_docs.py > full_docs_n200.txt 2>&1) &
(NTOK=100 REPS=40 ARCH=docs python run_docs.py > full_docs_n100.txt 2>&1; NTOK=100 REPS=40 ARCH=ctx python run_docs.py > full_ctx_n100.txt 2>&1) &
wait
python summarize.py > /dev/null
