#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python -m fets27_challenge.cli prepare-assets --data-root ./data/toy || \
  echo "Asset preparation skipped. Install PyTorch/MONAI/NiBabel first if you need toy data or checkpoints."

