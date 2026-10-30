#!/usr/bin/env bash
# One-command reproduction of the paper's IEMOCAP and CMU-MOSEI results.
#
#   ./scripts/download_and_eval.sh
#
# Requires: a CUDA GPU (~6 GB), the python environment from requirements.txt,
# and curl. MELD weights are not released; MELD training is run from scratch
# with the command in README.md ("Pipeline" section).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

echo '== 1/3 IEMOCAP: download weights + features (GitHub Release assets) =='
$PYTHON scripts/download_weights.py all

echo
echo '== 2/3 IEMOCAP: evaluate the released checkpoint =='
(cd IEMOCAP && $PYTHON multimodel_fusion.py --train False)
echo '   expected: [test] Acc: 73.81  WF1: 73.83'

echo
echo '== 3/3 CMU-MOSEI: verify the paper fusion recipe (no download, no training) =='
(cd MOSEI && $PYTHON verify_paper_fusion.py)
echo '   expected: test Acc-7: 55.0547'

echo
echo 'All checks finished.'
