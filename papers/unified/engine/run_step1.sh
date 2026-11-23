#!/bin/bash
cd /home/claude/paper/unified/engine
run() { STUDY72_T=$1 python3 wall81_bside.py $1 > study82_path/run_t$1.log 2>&1 && STUDY72_T=$1 python3 points81.py $1 >> study82_path/run_t$1.log 2>&1; echo "finished $1 $?"; }
(run 0.01525; run 0.01575; run 0.0165; run 0.017) &
(run 0.0155; run 0.016; run 0.01675; run 0.01725) &
wait; echo ALLDONE
