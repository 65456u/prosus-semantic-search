#!/bin/bash
# Generative retrieval trained up to 50 epochs (learning rate decays over 50), with early stopping after
# 10 epochs without a better Recall@10 on the full development set. A checkpoint is kept every 5 epochs
# and decoded on the 100 evaluation queries afterwards, for an nDCG-by-epoch curve (retrieval.generative.curve).
#SBATCH --job-name=prosus-gr50
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/gr50-%A_%a.out
set -euo pipefail
source scripts/snellius/twin.sh
VARIANTS=(tu-clean semantic semantic-mm)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
.venv-snellius/bin/python -m retrieval.generative.train --variant "$VARIANT" --epochs 50 --tag=-50ep --dev-sample 100000 --save-every 5 --patience 10
.venv-snellius/bin/python -m retrieval.generative.curve decode --variants "$VARIANT" --tag=-50ep
