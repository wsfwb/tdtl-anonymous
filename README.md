# TDTL / TDERC

Official implementation of **TDERC** (paper: *"Text-Dominant Information Flow for Multimodal Emotion Recognition in Conversation"*): text-dominant cross-modal interaction (WCMT), sequential branch update (TD), and text-guided auxiliary learning (TL) for multimodal emotion recognition.

**Results (paper Table 2 / Table 3):**

| Dataset | Metric | Score |
|---|---|---|
| IEMOCAP | Acc / WF1 | **73.81 / 73.83** |
| MELD | Acc / WF1 | **68.66 / 67.24** |
| CMU-MOSEI | Acc-7 | **55.05** |

Training logs reproducing these numbers are included in `paper_results/`.

## Repository layout

```
IEMOCAP/   full pipeline: stage-1 unimodal training, feature extraction, stage-2 fusion training
MELD/      stage-2 fusion training (features are built with the IEMOCAP stage-1 scripts)
MOSEI/     stage-2 fusion training + data conversion + fusion search
paper_results/  logs / configs / fusion recipes backing the reported numbers
```

## Environment

Python 3.10, PyTorch 2.7.1 (CUDA 12.8), transformers 5.0.0 (see `requirements.txt`). A single 24 GB GPU suffices for all experiments.

Pretrained feature extractors (frozen, downloaded from HuggingFace) are expected under `pretrained_model/`:

- `roberta-large`
- `data2vec-audio-base-960h`
- `timesformer-base-finetuned-k400` (first-stage video, IEMOCAP pipeline only)
- `videomae-base` (video feature extraction, IEMOCAP pipeline only)

```bash
mkdir -p pretrained_model
huggingface-cli download roberta-large --local-dir pretrained_model/roberta-large
huggingface-cli download facebook/data2vec-audio-base-960h --local-dir pretrained_model/data2vec-audio-base-960h
huggingface-cli download facebook/timesformer-base-finetuned-k400 --local-dir pretrained_model/timesformer-base-finetuned-k400
huggingface-cli download MCG-NJU/videomae-base --local-dir pretrained_model/videomae-base
```

## Data

### IEMOCAP

Place the dialogue-level CSV files under `datasets/IEMOCAP/` (columns `Utterance, Speaker, Emotion, Dialogue_ID, Utterance_ID, Wav_Path, Video_Path, Start_Time, End_Time`; train/dev/test split follows the standard 5-session protocol with 6 emotion classes: happy, sad, neutral, angry, excited, frustrated):

```
datasets/IEMOCAP/IEMOCAP_train.csv
datasets/IEMOCAP/IEMOCAP_dev.csv
datasets/IEMOCAP/IEMOCAP_test.csv
```

### MELD

Same CSV format as IEMOCAP, official splits:

```
datasets/MELD/MELD_train.csv
datasets/MELD/MELD_dev.csv
datasets/MELD/MELD_test.csv
```

### CMU-MOSEI

Download `unaligned_50.pkl` (MMSA / MultiBench processed MOSEI, e.g. from the [MMSA](https://github.com/thuiar/MMSA) releases) into `MOSEI/data/`, then convert:

```bash
cd MOSEI
python tools/convert_mosei_mmsa_to_magt.py --in_pkl data/unaligned_50.pkl --out_dir feature
```

## Pipeline

The pipeline has two stages. Stage 1 trains unimodal encoders and distills them into frozen 768-d utterance features (`<dataset>/feature/{train,dev,test}_features.pkl`); stage 2 trains the multimodal fusion model on those features.

### Stage 1 (IEMOCAP)

Run from the `IEMOCAP/` directory (MELD stage 1 uses the same scripts with the MELD CSVs):

```bash
# 1. unimodal teachers/students (each writes into ./IEMOCAP/save_model/)
python text.py            # -> text.bin
python audio.py           # -> audio.bin
python video.py           # -> video.bin
python KD.py              # -> text_KD_audio.bin, text_KD_video.bin  (text -> audio/video KD)

# 2. extract frozen features for stage 2 (writes feature/{train,dev,test}_features.pkl)
python extract_first_stage_features.py
```

`video_feature_extract.py` extracts raw VideoMAE frame features if you need the video front-end; the released features were produced with the pipeline above.

### Stage 2 (fusion training)

All stage-2 commands are run from the dataset directory and read `./feature/*_features.pkl`.

**IEMOCAP** (73.81 Acc / 73.83 WF1 — `paper_results/iemocap/winning_run.args`, log `paper_results/iemocap/multimodal_fusion_metrics_20260627_202304.json`, epoch 30):

```bash
cd IEMOCAP
python multimodel_fusion.py --train True --seed 42 --epochs 30 --batch_size 16 \
  --lr 1e-4 --l2 1e-6 --dropout 0.5 --n_rounds 1 --num_layers 6 \
  --loss_type cb_focal --cb_beta 0.9999 --focal_gamma 2.0 --label_smoothing 0 \
  --consistency_coef 0.1 --kd_a_w 0.5 --kd_v_w 0.5 \
  --pred_t_w 1.0 --pred_a_w 0.4 --pred_v_w 0.0
```

**MELD** (68.66 Acc / 67.24 WF1 — log `paper_results/meld/multimodal_fusion_metrics_20260322_211018.json`, epoch 19; the script defaults already carry the tuned configuration):

```bash
cd MELD
python multimodel_fusion.py            # defaults: seed 3407, 25 epochs, bs 32, lr 1e-4
```

**CMU-MOSEI** (55.05 Acc-7): the reported number is a weighted softmax fusion over 11 (run, epoch) checkpoints. The recipe and the member logits are included, so the number can be verified without retraining:

```bash
cd MOSEI
python verify_paper_fusion.py          # recomputes 55.0547 Acc-7 from paper_results/
```

## Evaluating a trained checkpoint

```bash
python multimodel_fusion.py --train False    # from the dataset directory; loads <DS>/<DS>/save_model/multimodal_fusion_best.bin
```

## Citation

```bibtex
@inproceedings{tdtl,
  title  = {Text-Dominant Information Flow for Multimodal Emotion Recognition in Conversation},
  author = {Anonymous Author(s)},
  year   = {2026}
}
```

## License

MIT.
