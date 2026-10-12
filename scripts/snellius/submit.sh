#!/bin/bash
# Submit a job array to both GPU partitions; each task keeps whichever copy starts first (see twin.sh).
#     bash scripts/snellius/submit.sh scripts/snellius/train50.sbatch [extra sbatch options, e.g. --array=1]
set -euo pipefail
SCRIPT=${1:?job script}
shift
RUN=$(basename "$SCRIPT" .sbatch)-$(date +%Y%m%d-%H%M%S)
mkdir -p logs
for partition in gpu_a100 gpu_h100; do
  job=$(sbatch --parsable -p "$partition" --export=ALL,RUN="$RUN" "$@" "$SCRIPT")
  echo "$partition $job" >> "logs/jobs-$RUN.txt.tmp"
done
mv "logs/jobs-$RUN.txt.tmp" "logs/jobs-$RUN.txt"
echo "run $RUN:"; cat "logs/jobs-$RUN.txt"
