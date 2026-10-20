# paper_results — evidence for the reported numbers

Each subdirectory holds the training logs / configs that back the paper's main
results. Nothing here was re-run; these are the original artifacts.

## iemocap/ — 73.81 Acc / 73.83 WF1

- `multimodal_fusion_metrics_20260627_202304.{json,csv}` — full 40-epoch
  training log of the reported run; epoch 30: `test_acc 73.81, test_fscore 73.83`.
- `winning_run.args` — exact command line (seed 42, `--pred_a_w 0.4`, ...).
- `pred_weight_grid_summary.tsv` — prediction-weight grid used to select
  `t1.0 / a0.4 / v0.0`.

## meld/ — 68.66 Acc / 67.24 WF1

- `multimodal_fusion_metrics_20260322_211018.{json,csv}` — training log;
  epoch 19: `test_acc 68.67, test_fscore 67.24` (script defaults = tuned config,
  seed 3407).

## mosei/ — 55.05 Acc-7

- `best_fusion_55p05.json` — fusion recipe: 11 (run, epoch) members, softmax
  temperature T=0.4, integer vote-count weights.
- `fusion_members/<run>/epoch_logits.json` — per-member text/audio/video logits
  for the recipe epochs (trimmed from the original 30-epoch files).
- `fusion_members/<run>/history.json` — full training args per member.
- `fusion_members/<run>/best.json` — per-member best-epoch labels/preds.
- `FINAL_RESULT.md` — search summary and the exact recipe table.

Reproduce the number without training: `cd MOSEI && python verify_paper_fusion.py`
(requires scipy + scikit-learn only, no GPU).
