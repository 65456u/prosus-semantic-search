#!/bin/bash
# Industry baseline: harrier fine-tuned as a two-tower retriever on search logs (InfoNCE, in-batch negatives).
#   0  search logs only            1  search logs + LLM-written queries per item
#SBATCH --job-name=prosus-2tower
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/twotower-%A_%a.out
set -euo pipefail
source scripts/snellius/twin.sh
DATA=(logs logs+synthetic)
.venv-snellius/bin/python -m retrieval.two_tower train --data "${DATA[$SLURM_ARRAY_TASK_ID]}" --epochs 5
