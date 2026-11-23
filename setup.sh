#!/usr/bin/env bash
# One-time setup. Run from the project folder in Terminal:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v conda >/dev/null; then
  for p in "$HOME/miniconda3" "$HOME/anaconda3"; do
    [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
  done
fi
command -v conda >/dev/null || { echo "conda not found. Install miniconda first."; exit 1; }
if conda env list | grep -qE '^atlaslens\s'; then
  conda env update -n atlaslens -f environment.yml
else
  conda env create -f environment.yml
fi
echo
echo "Optional deep-learning segmentation (large download, pulls PyTorch):"
echo "  conda run -n atlaslens pip install cellpose"
echo
echo "Next:"
echo "  conda activate atlaslens"
echo "  python tests/smoke_test.py         # checks the code on synthetic data (about 1 min)"
echo "  python -m atlaslens plan          # lists every file and its size, downloads nothing"
echo "  python -m atlaslens download      # asks for confirmation, then downloads"
echo "  python -m atlaslens scrna && python -m atlaslens merfish    # build the references"
echo "  python -m atlaslens.agent identify --structure DG            # run the agent (see README)"
