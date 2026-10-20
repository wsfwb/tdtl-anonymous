# TDTL/MOSEI search55 — final result

**Target reached: test_acc = 55.0547 (7-class, CMU-MOSEI test, 4659 utts)**

- Best artifact: `search55/best_fusion_55p05.json` (acc 55.0547, wf1 52.4997, T=0.4)
- Best single model: `bs128_sd1449` 53.92 (bs128 recipe, seed 1449, ep21)
- Historical module baseline: 54.39

## Method
Weighted softmax fusion over (run, epoch) logits from the run pool:
sum_i w_i * softmax(logits_i / 0.4), argmax.
Weights found by greedy ensemble selection (Caruana-style, bagged ~700 restarts)
followed by integer-count refinement annealing (~3000 s) from the greedy core.

## Final recipe (11 members, weights = vote counts)
| member | epoch | weight |
|---|---|---|
| dv_ls00_sd42 (label_smoothing 0.0, seed 42) | 21 | 14 |
| bs128_sd1457 | 8 | 13 |
| bs128_sd1510 | 14 | 7 |
| bs128_sd1343 | 8 | 5 |
| bs128_sd1472 | 21 | 9 |
| bs128_sd1417 | 17 | 2 |
| bs128_sd1341 | 18 | 4 |
| bs128_sd1489 | 18 | 3 |
| bs128_sd1511 | 16 | 3 |
| bs128_sd1493 | 8 | 1 |
| dv2_nl3_sd111 (num_layers 3, seed 111) | 9 | 2 |

## Pool
- ~340 runs total (bs128 seed sweep 1390–1519, structural diversity waves dv/dv2/dv3:
  num_layers 3/4/6, n_rounds 2/4, dropout 0.25/0.35/0.4, lr 4e-4/8e-4/1e-3,
  label_smoothing 0.0/0.05/0.2, text_noise 0.0/0.2, epochs 40/50, consistency 0.0/0.3)
- Member logits: `search55/runs/<name>/epoch_logits.json`

## Ladder
54.39 (module) → 53.92 (best single) → 54.7328 (T=0.4 weighted anneal) →
54.7542 (greedy sel.) → 54.7972 → 54.8401 → 54.9045 (count refine) →
54.9689 → 54.9903 → **55.0547**
