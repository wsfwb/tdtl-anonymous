"""Recompute the CMU-MOSEI Acc-7 reported in the paper (55.05) from the
released fusion recipe and member logits, without retraining.

Recipe (paper_results/mosei/best_fusion_55p05.json): weighted softmax fusion
over 11 (run, epoch) text-branch logits:

    probs = sum_i w_i * softmax(t_logits_i / T),  T = 0.4
    pred  = argmax(probs)

Weights are integer vote counts from greedy ensemble selection with
integer-count refinement (search55/anneal_full.py).

Run from the MOSEI/ directory:

    python verify_paper_fusion.py
"""
import json
import os

import numpy as np
from scipy.special import softmax as sp_softmax
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.join(HERE, "..", "paper_results", "mosei", "best_fusion_55p05.json")
MEMBERS = os.path.join(HERE, "..", "paper_results", "mosei", "fusion_members")


def main():
    with open(RECIPE) as f:
        recipe = json.load(f)

    T = recipe["T"]
    labels = None
    fused = None
    for run, epoch, weight in recipe["recipe"]:
        with open(os.path.join(MEMBERS, run, "epoch_logits.json")) as f:
            d = json.load(f)
        if labels is None:
            labels = np.array(d["labels"])
        t_logits = np.array(d[str(epoch)]["t_logits"], dtype=np.float32)
        probs = sp_softmax(t_logits / T, axis=1)
        contrib = weight * probs
        fused = contrib if fused is None else fused + contrib

    preds = fused.argmax(1)
    acc = (preds == labels).mean() * 100
    wf1 = f1_score(labels, preds, average="weighted") * 100
    print(f"members: {len(recipe['recipe'])}  T: {T}  n_test: {len(labels)}")
    print(f"test Acc-7: {acc:.4f}   (paper: 55.0547)")
    print(f"test WF1:   {wf1:.4f}   (paper: 52.4997)")


if __name__ == "__main__":
    main()
