#!/usr/bin/env bash
# One-command reproduction of the paper's IEMOCAP, MELD and CMU-MOSEI results.
#
#   ./scripts/download_and_eval.sh
#
# Requires: a CUDA GPU (~6 GB), the python environment from requirements.txt,
# and curl.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

echo '== 1/4 IEMOCAP: download weights + features (GitHub Release assets) =='
$PYTHON scripts/download_weights.py all

echo
echo '== 2/4 IEMOCAP: evaluate the released checkpoint =='
(cd IEMOCAP && $PYTHON multimodel_fusion.py --train False)
echo '   expected: [test] Acc: 73.81  WF1: 73.83'

echo
echo '== 3/4 MELD: evaluate the released checkpoint =='
(cd MELD && $PYTHON multimodel_fusion.py --train False)
echo '   expected: [test] Acc: 67.24  WF1: 68.66'

echo
echo '== 4/4 CMU-MOSEI: verify the paper fusion recipe'
(cd MOSEI && $PYTHON verify_paper_fusion.py)
echo '   expected: test Acc-7: 55.0547'

echo
echo 'All checks finished.'
