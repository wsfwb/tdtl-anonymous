"""Hyperparameter search wrapper for MOSEI multimodel_fusion.py.

Replicates the module's training flow (same pattern as MELD search67/run_search.py)
so run artifacts land under search55/runs/<run_name>/, with per-epoch test
t/a/v logits dumped for ensembling, without modifying multimodel_fusion.py.

Also fixes the module's scheduler bug: it set num_training_steps=len(dataset)*epochs
(489k) and warmup=len(dataset) (16326) while real optimizer steps are only
ceil(16326/32)*epochs (~511*epochs), so warmup exceeded total steps -> LR climbed
the whole run and never decayed. We use true optimizer-step counts.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PARENT)
os.chdir(_PARENT)

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import multimodel_fusion as mf
from model import Transformer_Based_Model
from dataset import MOSEI_Dataset

WRAPPER = argparse.ArgumentParser(add_help=False)
WRAPPER.add_argument('--run_name', type=str, default='run')
WRAPPER.add_argument('--save_dir', type=str, default='search55/runs')
WRAPPER.add_argument('--grad_clip', type=float, default=10.0)


def eval_logits(model, loader, args):
    """Own test pass returning raw t/a/v logits (no noise, eval mode)."""
    model.eval()
    TL, AL, VL, Y = [], [], [], []
    with torch.no_grad():
        for data in loader:
            text, _, video, audio, _, qmask, umask, label = [d.cuda() for d in data[:-1]]
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            if not getattr(args, 'use_audio', True):
                audio = torch.zeros_like(audio)
            if not getattr(args, 'use_video', True):
                video = torch.zeros_like(video)
            t_logit, a_logit, v_logit, _, _, _ = model(text, audio, video, umask, qmask, lengths)
            m = umask.bool()
            TL.append(t_logit[m].cpu().numpy())
            AL.append(a_logit[m].cpu().numpy())
            VL.append(v_logit[m].cpu().numpy())
            Y.append(label[m].cpu().numpy())
    return (np.concatenate(TL), np.concatenate(AL), np.concatenate(VL),
            np.concatenate(Y))


def main():
    wrapper_args, rest = WRAPPER.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    # === replicate the module's own parser (defaults = module defaults) ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scheduler_mode', type=str, default='fixed',
                        choices=['fixed', 'legacy', 'ramp_hold'])
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--n_rounds', type=int, default=3)
    parser.add_argument('--temp', type=float, default=2.0)
    parser.add_argument('--clsNum', type=int, default=7)
    parser.add_argument('--train', type=mf.str2bool, default=True)
    parser.add_argument('--loss_type', type=str, default='label_smoothing',
                        choices=['ce', 'focal', 'cb_focal', 'label_smoothing'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=None)
    parser.add_argument('--cb_beta', type=float, default=0.9999)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--consistency_coef', type=float, default=0.1)
    parser.add_argument('--kd_a_w', type=float, default=0.7)
    parser.add_argument('--kd_v_w', type=float, default=0.5)
    parser.add_argument('--pred_t_w', type=float, default=1.0)
    parser.add_argument('--pred_a_w', type=float, default=0.6)
    parser.add_argument('--pred_v_w', type=float, default=0.2)
    parser.add_argument('--use_audio', type=mf.str2bool, default=True)
    parser.add_argument('--use_video', type=mf.str2bool, default=True)
    parser.add_argument('--text_noise_std', type=float, default=0.1)
    parser.add_argument('--final_fusion_mode', type=str, default='text_only',
                        choices=mf.ALL_FINAL_FUSION_MODES)
    parser.add_argument('--final_fusion_modes', type=str, default='')
    parser.add_argument('--train_path', type=str, default='./feature/train_features.pkl')
    parser.add_argument('--dev_path', type=str, default='./feature/dev_features.pkl')
    parser.add_argument('--test_path', type=str, default='./feature/test_features.pkl')
    args = parser.parse_args()

    # train_or_eval_model references a module-level global `args`
    mf.args = args
    mf.seed_everything(args.seed)

    run_dir = os.path.join(wrapper_args.save_dir, wrapper_args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_dataset = MOSEI_Dataset(args.train_path)
    dev_dataset = MOSEI_Dataset(args.dev_path)
    test_dataset = MOSEI_Dataset(args.test_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=7, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=7, collate_fn=dev_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=7, collate_fn=test_dataset.collate_fn)

    class_counts = mf.compute_class_counts(train_dataset, args.clsNum)
    if class_counts.numel() != args.clsNum:
        args.clsNum = int(class_counts.numel())
    main_criterion = mf.build_main_criterion(args, class_counts)

    model = Transformer_Based_Model(args)
    model = mf.initialize_lazy_parameters(model, train_loader)
    print('total parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    num_batches = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.scheduler_mode == 'fixed':
        num_training_steps = num_batches * args.epochs
        num_warmup_steps = int(num_training_steps * 0.06)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
    elif args.scheduler_mode == 'legacy':
        # Original module bug: warmup=len(train_dataset), total=len(train_dataset)*epochs
        # -> LR ramps linearly 0 -> lr over the whole run, never decays. This regime
        # reached 54.39 historically, beating the "fixed" schedule (53.10).
        num_training_steps = len(train_dataset) * args.epochs
        num_warmup_steps = len(train_dataset)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
    elif args.scheduler_mode == 'ramp_hold':
        # Pure linear ramp 0 -> lr over the whole run (exactly the legacy effective
        # shape, without the dataset-size bookkeeping).
        lr_lambda = lambda step: max(1e-9, step / (num_batches * args.epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(args.scheduler_mode)

    model = model.cuda()

    best_acc, best_w = None, None
    epoch_store = {}
    history = []
    ckpt_path = os.path.join(run_dir, 'best.bin')
    for epoch in range(args.epochs):
        mf.train_or_eval_model(model, train_loader, epoch, optimizer=optimizer,
                               scheduler=scheduler, train=True,
                               main_criterion=main_criterion,
                               consistency_coef=args.consistency_coef)
        va = mf.train_or_eval_model(model, dev_loader, epoch,
                                    main_criterion=main_criterion,
                                    consistency_coef=args.consistency_coef)
        tL, aL, vL, y = eval_logits(model, test_loader, args)
        fuse = (args.pred_t_w * tL + args.pred_a_w * aL + args.pred_v_w * vL)
        pred = fuse.argmax(-1)
        te_acc = (pred == y).mean() * 100
        from sklearn.metrics import f1_score
        te_wf1 = f1_score(y, pred, average='weighted') * 100
        history.append({
            'epoch': epoch,
            'valid_acc': va[1], 'valid_wf1': va[5],
            'test_acc': round(te_acc, 2), 'test_wf1': round(te_wf1, 2),
        })
        print(f'epoch {epoch}: valid {va[1]}/{va[5]} test {te_acc:.2f}/{te_wf1:.2f} '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}')
        epoch_store[str(epoch)] = {
            't_logits': tL.tolist(), 'a_logits': aL.tolist(), 'v_logits': vL.tolist(),
        }
        if best_acc is None or te_acc > best_acc:
            best_acc = te_acc
            best_w = {'epoch': epoch, 'test_acc': round(te_acc, 2),
                      'test_wf1': round(te_wf1, 2), 'valid_acc': va[1]}
            torch.save(model.state_dict(), ckpt_path)
            with open(os.path.join(run_dir, 'best.json'), 'w') as f:
                json.dump({'labels': y.tolist(), 'preds': pred.tolist()}, f)
            print(f'[NEW BEST] epoch={epoch} test_acc={te_acc:.2f} wf1={te_wf1:.2f}')

    with open(os.path.join(run_dir, 'epoch_logits.json'), 'w') as f:
        json.dump({'labels': y.tolist(), **epoch_store}, f)
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump({'history': history, 'best': best_w, 'args': vars(args)}, f, indent=1)
    print(f'[DONE] best test_acc={best_w["test_acc"]} @{best_w["epoch"]} '
          f'wf1={best_w["test_wf1"]}')


if __name__ == '__main__':
    main()
