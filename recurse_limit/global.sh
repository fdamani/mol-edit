#!/bin/bash
INNERIND=6
for i in $(seq 1 $INNERIND)
do
    declare -a globalind=$i
    grep -rl globalind run_now_beam.sh | xargs sed 's/globalind/'$globalind'/g' | sbatch
done