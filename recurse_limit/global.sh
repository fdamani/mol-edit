#!/bin/bash
INNERIND=5
for i in $(seq 1 $INNERIND)
do
    # grep -rl ind run_now.sh | grep -rl j | xargs sed 's/globalind/'$globalind'/g' | sbatch
    declare -a globalind=$i
    grep -rl globalind run_now_beam.sh | xargs sed 's/globalind/'$globalind'/g' | sbatch
done
