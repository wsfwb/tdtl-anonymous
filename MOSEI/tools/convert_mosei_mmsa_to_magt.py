import os
import pickle
import numpy as np


def to_7class(y_any):
    """
    Convert MOSEI sentiment labels to 7-class indices [0..6].
    Supports:
      - regression float in [-3,3] (common): round -> clamp -> +3
      - already int -3..3: clamp -> +3
      - already int 0..6: keep
    """
    y = np.asarray(y_any).reshape(-1)

    # already 0..6 ints
    if np.issubdtype(y.dtype, np.integer) and y.min() >= 0 and y.max() <= 6:
        return y.astype(np.int64)

    # int -3..3 or float -3..3
    y_round = np.rint(y).astype(np.int64)
    y_round = np.clip(y_round, -3, 3)
    return (y_round + 3).astype(np.int64)


def pick_label_key(split: dict):
    """
    MMSA/MultiBench processed MOSEI may store labels under different keys.
    We try common candidates in order.
    """
    candidates = [
        "labels",
        "label",
        "y",
        "Y",
        "regression_labels",          # common for MOSEI sentiment regression
        "classification_labels",      # sometimes exists
        "sentiment",
        "sentiments",
    ]
    for k in candidates:
        if k in split:
            return k
    raise KeyError(f"Cannot find label key. Available keys: {list(split.keys())}")


def pick_vision_key(split: dict):
    candidates = ["vision", "video", "visual", "vis"]
    for k in candidates:
        if k in split:
            return k
    raise KeyError(f"Cannot find vision key. Available keys: {list(split.keys())}")


def build_dialogue_style(split, out_path):
    """
    Convert split dict to MAGTKD IEMOCAP_Dataset expected pkl format.
    Each sample is treated as a 1-utterance dialogue.
    """
    # text/audio keys are usually stable
    if "text" not in split:
        raise KeyError(f"Cannot find 'text'. Available keys: {list(split.keys())}")
    if "audio" not in split:
        raise KeyError(f"Cannot find 'audio'. Available keys: {list(split.keys())}")

    vkey = pick_vision_key(split)
    lkey = pick_label_key(split)

    text = split["text"]
    audio = split["audio"]
    vision = split[vkey]
    y = split[lkey]

    n = len(y)
    y7 = to_7class(y)

    out = {
        "text": {},
        "audio": {},
        "video": {},
        "audio_kd": {},
        "video_kd": {},
        "speakers": {},
        "labels": {},
        "vids": list(range(n)),
        "dia2utt": {},
    }

    for vid in range(n):
        out["dia2utt"][vid] = [0]

        out["text"][vid] = [np.asarray(text[vid])]
        out["audio"][vid] = [np.asarray(audio[vid])]
        out["video"][vid] = [np.asarray(vision[vid])]

        # no stage-1 KD => placeholder
        out["audio_kd"][vid] = [np.asarray(audio[vid])]
        out["video_kd"][vid] = [np.asarray(vision[vid])]

        out["speakers"][vid] = [0]
        out["labels"][vid] = [int(y7[vid])]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"[OK] wrote {out_path}  n={n}  label_key='{lkey}'  vision_key='{vkey}'")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pkl", type=str, required=True, help="unaligned_50.pkl")
    ap.add_argument("--out_dir", type=str, default="./feature", help="output feature dir")
    args = ap.parse_args()

    with open(args.in_pkl, "rb") as f:
        data = pickle.load(f)

    # show keys once (useful if anything mismatches)
    print("Top-level keys:", list(data.keys()))
    print("Train keys:", list(data["train"].keys()))

    valid_key = "valid" if "valid" in data else ("dev" if "dev" in data else None)
    if valid_key is None:
        raise KeyError(f"Cannot find valid/dev split keys in: {list(data.keys())}")

    build_dialogue_style(data["train"], os.path.join(args.out_dir, "first_stage_train_features.pkl"))
    build_dialogue_style(data[valid_key], os.path.join(args.out_dir, "first_stage_dev_features.pkl"))
    build_dialogue_style(data["test"], os.path.join(args.out_dir, "first_stage_test_features.pkl"))


if __name__ == "__main__":
    main()