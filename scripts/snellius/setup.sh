#!/bin/bash
# One-time Python environment on scratch (home directories are small). Runs on the login node.
set -euo pipefail
module load 2025 Python/3.13.1-GCCcore-14.2.0
export XDG_CACHE_HOME="$PWD/.cache" PIP_CACHE_DIR="$PWD/.cache/pip"
if [ ! -x .venv-snellius/bin/python ]; then
  python -m venv .venv-snellius
  .venv-snellius/bin/pip install -q --upgrade pip
  .venv-snellius/bin/pip install -q -r requirements.lock.txt
fi
.venv-snellius/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
