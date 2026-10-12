#!/bin/zsh
# Fetch the cluster checkpoints (tu-raw and the -long runs) and training curves, then decode on the Mac:
#     SNELLIUS=<user>@snellius.surf.nl scripts/snellius/pull.sh
#     .venv/bin/python -m retrieval.generative.run --variants tu-raw
#     .venv/bin/python -m retrieval.generative.run --variants tu-clean semantic semantic-mm --tag=-long
# Checkpoints trained on the laptop are never overwritten.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SNELLIUS:?set SNELLIUS=<user>@snellius.surf.nl}"
REMOTE_DIR=${REMOTE_DIR:-/scratch-shared/${SNELLIUS%@*}/prosus}
rsync -az --include='tu-raw/***' --include='*-long/***' --exclude='*' "$SNELLIUS:$REMOTE_DIR/.demo-cache/genret/" .demo-cache/genret/
rsync -az --include='train-tu-raw.json' --include='train-*-long.json' --exclude='*' "$SNELLIUS:$REMOTE_DIR/output/genret/" output/genret/
rsync -az "$SNELLIUS:$REMOTE_DIR/logs/" .demo-cache/runtime/snellius-logs/
