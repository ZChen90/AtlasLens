#!/bin/bash
# Double-click this file in Finder to download the datasets (21.3 GB) into data/.
# It can be stopped and restarted; finished files are kept and partial ones resume.
cd "$(dirname "$0")"
PY=""
for c in "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" "$(command -v python3)"; do
  [ -x "$c" ] && PY="$c" && break
done
[ -z "$PY" ] && { echo "No Python found. Install miniconda first."; read -r -p "Press Return to close."; exit 1; }
"$PY" atlaslens/fetch.py --yes
echo
read -r -p "Finished. Press Return to close this window."
