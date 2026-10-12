#!/bin/bash
# Generative-retrieval training, one GPU per array task. submit.sh submits this array to gpu_a100
# and gpu_h100 at once; whichever copy of a task starts first takes a lock and cancels its twin.
#   0    tu-raw, equal budget with the laptop runs (epoch 1 of a 2-epoch schedule): completes the ID ablation
#   1-3  tu-clean, semantic, semantic-mm trained longer (tag -long): does more training close the gap?
# A walltime of at most one hour uses Snellius's short-job nodes, which start sooner.
#SBATCH --job-name=prosus-gr
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/gr-%A_%a.out
set -euo pipefail
EPOCHS=${EPOCHS:-8}
source scripts/snellius/twin.sh
JOBS=("--variant tu-raw --epochs 2 --stop-after-epoch 1 --dev-sample 200"
      "--variant tu-clean --epochs $EPOCHS --tag=-long --dev-sample 200"
      "--variant semantic --epochs $EPOCHS --tag=-long --dev-sample 200"
      "--variant semantic-mm --epochs $EPOCHS --tag=-long --dev-sample 200")
.venv-snellius/bin/python -m retrieval.generative.train ${JOBS[$SLURM_ARRAY_TASK_ID]}
