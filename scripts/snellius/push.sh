#!/bin/zsh
# Copy what generative-retrieval training needs to a private scratch directory on Snellius and
# build the Python environment there once. Then, on Snellius: bash scripts/snellius/submit.sh
#
#     SNELLIUS=<user>@snellius.surf.nl scripts/snellius/push.sh
#
# Only code, the two assignment CSVs, the prepared identifiers, and the T5Gemma 2 weights are
# copied; the directory is chmod 700. Pull results back with pull.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SNELLIUS:?set SNELLIUS=<user>@snellius.surf.nl}"
REMOTE_DIR=${REMOTE_DIR:-/scratch-shared/${SNELLIUS%@*}/prosus}
ssh "$SNELLIUS" "mkdir -p $REMOTE_DIR/logs $REMOTE_DIR/output/genret $REMOTE_DIR/.demo-cache/models && chmod 700 $REMOTE_DIR"
rsync -az --exclude __pycache__ core retrieval evaluation scripts data requirements.lock.txt "$SNELLIUS:$REMOTE_DIR/"
rsync -az output/genret/items.jsonl output/genret/ids.json "$SNELLIUS:$REMOTE_DIR/output/genret/"
rsync -az .demo-cache/models/t5gemma-2-270m-270m "$SNELLIUS:$REMOTE_DIR/.demo-cache/models/"
ssh "$SNELLIUS" "cd $REMOTE_DIR && bash scripts/snellius/setup.sh"
