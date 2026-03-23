import numpy as np
import argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import pickle as pk
import datetime
import os
import random
import torch.nn as nn
import json
import csv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model import Transformer_Based_Model, MaskedKLDivLoss, MaskedNLLLoss, TestModel, Fusion_model
from dataset import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_FINAL_FUSION_MODES = [
    'text_only',
    'mean',
    'weighted_sum',
    'concat_linear',
    'gate_concat',
    'adaptive_fusion',
]


def resolve_feature_path(path_str):
    if os.path.exists(path_str):
        return path_str
    normalized = path_str.lstrip('./').lstrip('/')
    candidate = os.path.join(BASE_DIR, normalized)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f'Feature file not found: {path_str}')


def sanitize_tag(tag):
    return str(tag).replace('/', '_').replace(' ', '_')


def parse_final_fusion_modes(final_fusion_mode, final_fusion_modes):
    if final_fusion_modes is None or str(final_fusion_modes).strip() == '':
        return [final_fusion_mode]

    raw = str(final_fusion_modes).strip()
    if raw.lower() == 'all':
        return list(ALL_FINAL_FUSION_MODES)

    modes = [x.strip() for x in raw.split(',') if x.strip()]
    invalid = [m for m in modes if m not in ALL_FINAL_FUSION_MODES]
    if invalid:
        raise ValueError(f'Invalid final fusion modes: {invalid}. Supported: {ALL_FINAL_FUSION_MODES}')
    return modes


def get_experiment_name(args):
    return f'finalfusion_{sanitize_tag(args.final_fusion_mode)}'


def get_checkpoint_path(save_dir, exp_name):
    return os.path.join(save_dir, f'{exp_name}_best.bin')


def get_pred_json_path(save_dir, exp_name):
    return os.path.join(save_dir, f'{exp_name}_best.json')




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

class Logit_Loss(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=2.0):
        super(Logit_Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss

class Feature_Loss(nn.Module):
    def __init__(self, temp=1.0):
        super(Feature_Loss, self).__init__()
        self.t = temp

    def forward(self, other_embd, text_embd):
        text_embd = F.normalize(text_embd, p=2, dim=1)
        other_embd = F.normalize(other_embd, p=2, dim=1)
        target = torch.matmul(text_embd, text_embd.transpose(0,1))
        x = torch.matmul(text_embd, other_embd.transpose(0,1))
        log_q = torch.log_softmax(x / self.t, dim=1)
        p = torch.softmax(target / self.t, dim=1)
        return F.kl_div(log_q, p, reduction='batchmean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=1)
        prob = log_prob.exp()
        ce_loss = -log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - prob.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            ce_loss = ce_loss * alpha_t
        loss = focal_factor * ce_loss
        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return loss.sum()
        return loss.mean()


def compute_class_balanced_alpha(class_counts, beta):
    effective_num = 1.0 - torch.pow(beta, class_counts)
    alpha = (1.0 - beta) / (effective_num + 1e-8)
    alpha = alpha / alpha.sum() * len(alpha)
    return alpha


def build_main_criterion(args, class_counts=None):
    if args.loss_type == 'label_smoothing':
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.loss_type in ['focal', 'cb_focal']:
        alpha = None
        if args.loss_type == 'cb_focal' and class_counts is not None:
            alpha = compute_class_balanced_alpha(class_counts.float(), args.cb_beta)
        elif args.focal_alpha is not None:
            alpha = torch.tensor([args.focal_alpha] * args.clsNum)
        return FocalLoss(alpha=alpha, gamma=args.focal_gamma)

    return nn.CrossEntropyLoss()


def symmetric_kl_divergence(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=1)
    log_q = F.log_softmax(logits_q, dim=1)
    p = log_p.exp()
    q = log_q.exp()
    return F.kl_div(log_p, q, reduction='batchmean') + F.kl_div(log_q, p, reduction='batchmean')


def compute_class_counts(dataset, cls_num):
    counts = torch.zeros(cls_num)
    for label_list in dataset.labels.values():
        binc = torch.bincount(torch.tensor(label_list), minlength=cls_num)
        if binc.numel() > counts.numel():
            new_counts = torch.zeros(binc.numel())
            new_counts[:counts.numel()] = counts
            counts = new_counts
        counts[:binc.numel()] += binc
    return counts


def initialize_lazy_parameters(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        text, audio, video, audio_kd, video_kd, qmask, umask, _ = [d.to(device) for d in batch[:-1]]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        model(text, audio_kd, video, umask, qmask, lengths)
    return model

def CE_Loss(args, pred_outs, logit_t, hidden_s, hidden_t, labels):
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)
    logit_loss = Logit_Loss().cuda()
    logit_loss = logit_loss(pred_outs, logit_t)
    feature_loss = Feature_Loss().cuda()
    feature_loss = feature_loss(hidden_s, hidden_t)
    loss_val = ori_loss + 0.1*logit_loss + feature_loss
    return loss_val

def train_or_eval_model(model, data_loader, epoch, optimizer=None, scheduler=None, train=False, main_criterion=None, consistency_coef=0.0, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    losses, preds, labels, masks = [], [], [], []
    losses_a_kd, losses_v_kd = [], []

    assert not train or optimizer!=None
    model.cuda()
    base_ce = nn.CrossEntropyLoss()
    if train:
        model.train()
    else:
        model.eval()

    for data in data_loader:
        if train:
            optimizer.zero_grad()
        text, audio, video, audio_kd, video_kd, qmask, umask, label = [d.cuda() for d in data[:-1]]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # --- modality ablation (input-level): T-only / T+A / T+V ---
        # NOTE: this project feeds audio_kd as audio input, and video (NOT video_kd) as video input
        if not getattr(args, 'use_audio', True):
            audio_kd = torch.zeros_like(audio_kd)
        if not getattr(args, 'use_video', True):
            video = torch.zeros_like(video)

        # t_logit, a_logit, v_logit, t_hidden, a_hidden, v_hidden = model(text, audio_kd, video_kd, umask, qmask, lengths)
        t_logit, a_logit, v_logit, t_hidden, a_hidden, v_hidden = model(text, audio_kd, video, umask, qmask, lengths)
        # _, t_logit = model(text, audio, video)


        umask_bool = umask.bool()
        labels_ = label[umask_bool]

        logit_t = t_logit[umask_bool]
        logit_a = a_logit[umask_bool]
        logit_v = v_logit[umask_bool]

        hidden_t = t_hidden[umask_bool]
        hidden_a = a_hidden[umask_bool]
        hidden_v = v_hidden[umask_bool]

        loss_kd_a = CE_Loss(args, logit_a, logit_t, hidden_a, hidden_t, labels_)
        loss_kd_v = CE_Loss(args, logit_v, logit_t, hidden_v, hidden_t, labels_)


        # loss_val = loss(logit_t , labels_) + 0.5 * (loss_kd_a + loss_kd_v)
        a = getattr(args, 'kd_a_w', 0.7)
        b = getattr(args, 'kd_v_w', 0.8)
        main_loss = main_criterion(logit_t, labels_) if main_criterion is not None else base_ce(logit_t, labels_)
        loss_val = main_loss + a * loss_kd_a + b * loss_kd_v
        # loss_val = main_loss

        if consistency_coef > 0:
            cons_loss = (
                symmetric_kl_divergence(logit_t, logit_a) +
                symmetric_kl_divergence(logit_t, logit_v) +
                symmetric_kl_divergence(logit_a, logit_v)
            )
            loss_val = loss_val + consistency_coef * cons_loss

        pred_ = torch.argmax(logit_t , dim=1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss_val.item())
        losses_a_kd.append(loss_kd_a.item())
        losses_v_kd.append(loss_kd_v.item())
        if train:
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()
    if preds!=[]:
        preds = np.concatenate(preds)
        masks = np.concatenate(masks)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_loss_a_kd = round(np.sum(losses_a_kd) / len(losses_a_kd), 4)
    avg_loss_v_kd = round(np.sum(losses_v_kd) / len(losses_v_kd), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    mask2 = labels != 3
    if np.any(mask2):
        labels2 = labels[mask2]
        preds2 = preds[mask2]
        labels2_bin = (labels2 > 3).astype(int)
        preds2_bin = (preds2 > 3).astype(int)
        avg_acc2 = round(accuracy_score(labels2_bin, preds2_bin) * 100, 2)
        avg_f1_bin = round(f1_score(labels2_bin, preds2_bin, average='binary') * 100, 2)
    else:
        avg_acc2 = float('nan')
        avg_f1_bin = float('nan')

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, avg_acc2, avg_f1_bin, avg_loss_a_kd, avg_loss_v_kd


def _SaveModel(model, save_path, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, model_name))


def save_labels_and_preds(labels, preds, filename):
    data = {
        'labels': labels.tolist(),
        'preds': preds.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

@torch.no_grad()
def collect_tsne_features(model, data_loader, args):
    model.eval()
    model.cuda()

    feats = {
        'text': [],
        'audio': [],
        'video': [],
        'text_tl_audio': [],
        'text_tl_video': [],
    }
    labels_all = []

    for data in data_loader:
        text, audio, video, audio_kd, video_kd, qmask, umask, label = [d.cuda() for d in data[:-1]]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if not getattr(args, 'use_audio', True):
            audio_kd = torch.zeros_like(audio_kd)
        if not getattr(args, 'use_video', True):
            video = torch.zeros_like(video)

        out = model(text, audio_kd, video, umask, qmask, lengths, return_features=True)

        batch_size = out['text'].size(0)
        seq_len = out['text'].size(1)
        valid_mask = umask.bool()

        for key in feats.keys():
            x = out[key].reshape(batch_size * seq_len, -1)
            m = valid_mask.reshape(batch_size * seq_len)
            feats[key].append(x[m].detach().cpu().numpy())

        y = label.reshape(batch_size * seq_len)
        y = y[valid_mask.reshape(batch_size * seq_len)]
        labels_all.append(y.detach().cpu().numpy())

    for k in feats:
        feats[k] = np.concatenate(feats[k], axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return feats, labels_all


def balanced_sample_per_class(x, y, max_per_class=200, seed=42):
    rng = np.random.RandomState(seed)
    keep_idx = []
    for c in sorted(np.unique(y).tolist()):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep_idx.extend(idx.tolist())
    keep_idx = np.array(keep_idx)
    return x[keep_idx], y[keep_idx]


def run_and_plot_tsne(model, data_loader, args):
    feats, labels = collect_tsne_features(model, data_loader, args)
    class_names = ['strong_neg', 'neg', 'weak_neg', 'neutral', 'weak_pos', 'pos', 'strong_pos']
    plot_order = ['text', 'audio', 'text_tl_audio', 'video', 'text_tl_video']
    plot_titles = ['(a)text', '(b)audio', '(c)text_tl_audio', '(d)video', '(e)text_tl_video']

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    for ax, key, title in zip(axes, plot_order, plot_titles):
        x, y = balanced_sample_per_class(feats[key], labels, max_per_class=args.tsne_max_per_class, seed=args.seed)
        emb = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=args.seed).fit_transform(x)
        for c in sorted(np.unique(y).tolist()):
            idx = (y == c)
            ax.scatter(emb[idx, 0], emb[idx, 1], s=4, alpha=0.7, label=class_names[c])
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center right', fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    os.makedirs(os.path.dirname(args.tsne_save), exist_ok=True)
    plt.savefig(args.tsne_save, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[TSNE] saved to {args.tsne_save}')

def model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, args, main_criterion, consistency_coef, exp_name):
    best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    history = []
    save_dir = './MOSEI/save_model'
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = get_checkpoint_path(save_dir, exp_name)
    best_pred_path = get_pred_json_path(save_dir, exp_name)

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc, _, _, _, train_fscore, train_acc2, train_f1, train_loss_a_kd, train_loss_v_kd = train_or_eval_model(model, train_loader, epoch, optimizer, scheduler, True, main_criterion, consistency_coef)
        valid_loss, valid_acc, _, _, _, valid_fscore, valid_acc2, valid_f1, valid_loss_a_kd, valid_loss_v_kd = train_or_eval_model(model, dev_loader, epoch, main_criterion=main_criterion, consistency_coef=consistency_coef)
        test_loss, test_acc, label, pred, _, test_fscore, test_acc2, test_f1, test_loss_a_kd, test_loss_v_kd = train_or_eval_model(model, test_loader, epoch, main_criterion=main_criterion, consistency_coef=consistency_coef)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_fscore': test_fscore,
            'test_acc2': test_acc2,
            'test_f1': test_f1,
        })

        print(f'epoch: {epoch}, train_loss: {train_loss}, train_acc7: {train_acc}, train_wf1: {train_fscore}, train_acc2: {train_acc2}, train_f1: {train_f1} | valid_loss: {valid_loss}, valid_acc7: {valid_acc}, valid_wf1: {valid_fscore}, valid_acc2: {valid_acc2}, valid_f1: {valid_f1} | test_loss: {test_loss}, test_acc7: {test_acc}, test_wf1: {test_fscore}, test_acc2: {test_acc2}, test_f1: {test_f1}, time: {time.time()}')
        print(f'epoch: {epoch}, train_loss_a_kd: {train_loss_a_kd}, train_loss_v_kd: {train_loss_v_kd}, valid_loss_a_kd: {valid_loss_a_kd}, valid_loss_v_kd: {valid_loss_v_kd}, test_loss_a_kd: {test_loss_a_kd}, test_loss_v_kd: {test_loss_v_kd}')

        if best_acc == None or test_acc > best_acc:
            prev_best = -1 if best_acc is None else best_acc
            best_acc = test_acc
            print(f'[NEW BEST] epoch={epoch}  test_acc={best_acc}  (prev_best={prev_best})')
            _SaveModel(model, save_dir, os.path.basename(best_ckpt_path))
            save_labels_and_preds(label, pred, best_pred_path)
            print(classification_report(label, pred, digits=4, zero_division=0))
            print(f'done')

    print('\n=== Epoch Summary ===')
    print('epoch\ttrain_loss\ttrain_acc7\tvalid_loss\tvalid_acc7\ttest_loss\ttest_acc7\ttest_wf1\ttest_acc2\ttest_f1')
    for h in history:
        print(
            f"{h['epoch']}\t{h['train_loss']}\t{h['train_acc']}\t{h['valid_loss']}\t{h['valid_acc']}\t{h['test_loss']}\t{h['test_acc']}\t{h['test_fscore']}\t{h['test_acc2']}\t{h['test_f1']}"
        )

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(save_dir, f'{exp_name}_metrics_{timestamp}.csv')
    json_path = os.path.join(save_dir, f'{exp_name}_metrics_{timestamp}.json')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'epoch',
                'train_loss', 'train_acc',
                'valid_loss', 'valid_acc',
                'test_loss', 'test_acc', 'test_fscore',
                'test_acc2', 'test_f1'
            ]
        )
        writer.writeheader()
        writer.writerows(history)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    print(f'[Saved] metrics csv: {csv_path}')
    print(f'[Saved] metrics json: {json_path}')

    return {
        'exp_name': exp_name,
        'final_fusion_mode': args.final_fusion_mode,
        'best_acc': best_acc,
        'best_checkpoint': best_ckpt_path,
        'best_pred_json': best_pred_path,
    }


def run_single_experiment(args, train_loader, dev_loader, test_loader, train_dataset):
    seed_everything(args.seed)

    class_counts = compute_class_counts(train_dataset, args.clsNum)
    if class_counts.numel() != args.clsNum:
        print(f'[INFO] clsNum adjusted from {args.clsNum} to {class_counts.numel()} based on label ids in dataset.')
        args.clsNum = int(class_counts.numel())
    main_criterion = build_main_criterion(args, class_counts)

    model = Transformer_Based_Model(args)
    model = initialize_lazy_parameters(model, train_loader)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    exp_name = get_experiment_name(args)
    save_dir = './MOSEI/save_model'
    ckpt_path = get_checkpoint_path(save_dir, exp_name)

    if not args.train:
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
        else:
            print(f'[WARN] checkpoint not found: {ckpt_path}, switch to train mode.')
            args.train = True

    if args.draw_tsne:
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
        else:
            raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
        if args.tsne_split == 'train':
            tsne_loader = train_loader
        elif args.tsne_split == 'dev':
            tsne_loader = dev_loader
        else:
            tsne_loader = test_loader

        root, ext = os.path.splitext(args.tsne_save)
        if ext == '':
            ext = '.png'
        args.tsne_save = f'{root}_{sanitize_tag(args.final_fusion_mode)}{ext}'
        run_and_plot_tsne(model, tsne_loader, args)
        return {
            'exp_name': exp_name,
            'final_fusion_mode': args.final_fusion_mode,
            'tsne_save': args.tsne_save,
        }

    num_training_steps = len(train_dataset) * args.epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    return model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, args, main_criterion, args.consistency_coef, exp_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training.')
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 regularization weight.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training.')
    parser.add_argument('--seed', type=int, default=42, help='random seed for training.')
    parser.add_argument('--epochs', type=int, default=30, help='epoch for training.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden dimension.')
    parser.add_argument('--n_head', type=int, default=8, help='number of heads.')
    parser.add_argument('--num_layers', type=int, default=6, help='number of heads.')
    parser.add_argument('--n_rounds', type=int, default=1, help='number of interaction rounds for Transformer Model.')
    parser.add_argument('--temp', type=float, default=2.0, help='temperature for contrastive learning.')
    parser.add_argument('--clsNum', type=int, default=7, help='number of classes.')
    parser.add_argument('--train', type=str2bool, default=True, help='whether to train the model.')
    parser.add_argument('--loss_type', type=str, default='cb_focal', choices=['ce', 'focal', 'cb_focal', 'label_smoothing'], help='main classification loss type.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for focal loss.')
    parser.add_argument('--focal_alpha', type=float, default=None, help='scalar alpha for focal loss (overridden by class-balanced).')
    parser.add_argument('--cb_beta', type=float, default=0.9999, help='beta for class-balanced focal loss.')
    parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing factor for CE.')
    parser.add_argument('--consistency_coef', type=float, default=0.1, help='weight for symmetric KL consistency between modalities.')
    parser.add_argument('--kd_a_w', type=float, default=0.5, help='weight for audio KD loss (student=audio, teacher=text).')
    parser.add_argument('--kd_v_w', type=float, default=0.5, help='weight for video KD loss (student=video, teacher=text).')
    parser.add_argument('--use_audio', type=str2bool, default=True, help='whether to use audio modality input (audio_kd).')
    parser.add_argument('--use_video', type=str2bool, default=True, help='whether to use video modality input (video).')
    parser.add_argument('--final_fusion_mode', type=str, default='text_only', choices=ALL_FINAL_FUSION_MODES, help='final fusion strategy applied to [final_transformer_out, a_transformer_out, v_transformer_out].')
    parser.add_argument('--final_fusion_modes', type=str, default='', help='comma-separated final fusion modes to run in batch, or "all". Empty means only use --final_fusion_mode.')
    parser.add_argument('--train_path', type=str, default='/feature/train_features.pkl', help='path to training pkl feature file.')
    parser.add_argument('--dev_path', type=str, default='/feature/dev_features.pkl', help='path to dev pkl feature file.')
    parser.add_argument('--test_path', type=str, default='/feature/test_features.pkl', help='path to test pkl feature file.')
    parser.add_argument('--draw_tsne', type=str2bool, default=False, help='draw t-SNE on CMU-MOSEI')
    parser.add_argument('--tsne_max_per_class', type=int, default=200, help='max samples per class for t-SNE')
    parser.add_argument('--tsne_split', type=str, default='test', choices=['train', 'dev', 'test'], help='which split to use for t-SNE')
    parser.add_argument('--tsne_save', type=str, default='./MOSEI/save_model/mosei_tsne_5views.png', help='path to save t-SNE figure')
    args = parser.parse_args()

    # create dataloader
    train_path = resolve_feature_path(args.train_path)
    dev_path = resolve_feature_path(args.dev_path)
    test_path = resolve_feature_path(args.test_path)

    train_dataset = MOSEI_Dataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=7, collate_fn=train_dataset.collate_fn)

    dev_dataset = MOSEI_Dataset(dev_path)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=dev_dataset.collate_fn)

    test_dataset = MOSEI_Dataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=test_dataset.collate_fn)

    modes_to_run = parse_final_fusion_modes(args.final_fusion_mode, args.final_fusion_modes)
    all_results = []

    for mode in modes_to_run:
        print('\n' + '#' * 80)
        print(f'Running final fusion mode: {mode}')
        print('#' * 80)
        args.final_fusion_mode = mode
        result = run_single_experiment(args, train_loader, dev_loader, test_loader, train_dataset)
        all_results.append(result)

    if (not args.draw_tsne) and all_results:
        print('\n' + '=' * 72)
        print('Final fusion summary')
        print('=' * 72)
        for item in all_results:
            print(f"mode={item['final_fusion_mode']}\tbest_acc={item['best_acc']}\tckpt={item['best_checkpoint']}")

        save_dir = './MOSEI/save_model'
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_json = os.path.join(save_dir, f'final_fusion_summary_{ts}.json')
        summary_csv = os.path.join(save_dir, f'final_fusion_summary_{ts}.csv')

        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['exp_name', 'final_fusion_mode', 'best_acc', 'best_checkpoint', 'best_pred_json'])
            writer.writeheader()
            writer.writerows(all_results)

        print(f'[Summary saved] {summary_json}')
        print(f'[Summary saved] {summary_csv}')


