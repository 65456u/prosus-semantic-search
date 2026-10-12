# Sourced by every job script: the first copy of an array task to start (on gpu_a100 or gpu_h100)
# takes a lock and cancels its still-pending twin on the other partition.
cd "$SLURM_SUBMIT_DIR"
RUN=${RUN:?submit with submit.sh}
for _ in $(seq 60); do [ -f "logs/jobs-$RUN.txt" ] && break; sleep 2; done
mkdir -p logs/locks
if ! mkdir "logs/locks/$RUN-${SLURM_ARRAY_TASK_ID:-0}" 2>/dev/null; then
  echo "task ${SLURM_ARRAY_TASK_ID:-0} already runs on the other partition"; exit 0
fi
while read -r partition job; do
  if [ "$partition" != "$SLURM_JOB_PARTITION" ]; then scancel "${job}_${SLURM_ARRAY_TASK_ID:-0}" 2>/dev/null || true; fi
done < "logs/jobs-$RUN.txt"
echo "task ${SLURM_ARRAY_TASK_ID:-0} on $SLURM_JOB_PARTITION: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
module load 2025 Python/3.13.1-GCCcore-14.2.0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false XDG_CACHE_HOME="$PWD/.cache" TRITON_CACHE_DIR="$PWD/.cache/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
