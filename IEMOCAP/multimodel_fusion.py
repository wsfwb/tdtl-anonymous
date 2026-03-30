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

from model import Transformer_Based_Model, MaskedKLDivLoss, MaskedNLLLoss, TestModel, Fusion_model
from dataset import *

from sklearn.manifold import TSNE




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
        counts += torch.bincount(torch.tensor(label_list), minlength=cls_num)
    return counts

def CE_Loss(args, pred_outs, logit_t, hidden_s, hidden_t, labels):
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)
    logit_loss = Logit_Loss().cuda()
    logit_loss = logit_loss(pred_outs, logit_t)
    feature_loss = Feature_Loss().cuda()
    feature_loss = feature_loss(hidden_s, hidden_t)
    loss_val = ori_loss + 0.1*logit_loss + feature_loss
    return loss_val


def get_tsne_loader(args, split):
    if split == 'train':
        ds = IEMOCAP_Dataset('./feature/train_features.pkl')
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=ds.collate_fn)
        return ds, dl
    if split == 'dev':
        ds = IEMOCAP_Dataset('./feature/dev_features.pkl')
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=ds.collate_fn)
        return ds, dl
    if split == 'test':
        ds = IEMOCAP_Dataset('./feature/test_features.pkl')
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=ds.collate_fn)
        return ds, dl
    raise ValueError(f'Unknown tsne split: {split}')


def collect_tsne_features(model, data_loader, args, tsne_mode='text', max_per_class=200):
    model.eval()
    model.cuda()

    class_names = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    feats_by_class = {i: [] for i in range(args.clsNum)}

    with torch.no_grad():
        for data in data_loader:
            text, _, video, audio, _, qmask, umask, label = [d.cuda() for d in data[:-1]]
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

            if not getattr(args, 'use_audio', True):
                audio = torch.zeros_like(audio)
            if not getattr(args, 'use_video', True):
                video = torch.zeros_like(video)

            outputs = model(text, audio, video, umask, qmask, lengths, return_features=True)
            _, _, _, _, _, _, feature_dict = outputs

            if tsne_mode not in feature_dict:
                raise ValueError(f'Unsupported tsne_mode={tsne_mode}, available={list(feature_dict.keys())}')

            feat = feature_dict[tsne_mode]
            umask_bool = umask.bool()
            labels_ = label[umask_bool]
            feat_ = feat[umask_bool]

            for c in range(args.clsNum):
                if len(feats_by_class[c]) >= max_per_class:
                    continue
                idx = (labels_ == c).nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                remain = max_per_class - len(feats_by_class[c])
                idx = idx[:remain]
                feats_by_class[c].append(feat_[idx].detach().cpu())

            if all(len(feats_by_class[c]) >= max_per_class for c in range(args.clsNum)):
                break

    all_x, all_y = [], []
    for c in range(args.clsNum):
        if len(feats_by_class[c]) == 0:
            continue
        x_c = torch.cat(feats_by_class[c], dim=0)
        y_c = torch.full((x_c.size(0),), c, dtype=torch.long)
        all_x.append(x_c)
        all_y.append(y_c)

    if len(all_x) == 0:
        raise RuntimeError(f'No features collected for tsne_mode={tsne_mode}')

    X = torch.cat(all_x, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    return X, y, class_names


def collect_tsne_features_all(model, data_loader, args, max_per_class=200):
    model.eval()
    model.cuda()

    class_names = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    mode_names = ['text', 'audio', 'video', 'text_tl_audio', 'text_tl_video']
    feats_by_mode_class = {
        mode: {i: [] for i in range(args.clsNum)}
        for mode in mode_names
    }

    with torch.no_grad():
        for data in data_loader:
            text, _, video, audio, _, qmask, umask, label = [d.cuda() for d in data[:-1]]
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

            if not getattr(args, 'use_audio', True):
                audio = torch.zeros_like(audio)
            if not getattr(args, 'use_video', True):
                video = torch.zeros_like(video)

            outputs = model(text, audio, video, umask, qmask, lengths, return_features=True)
            _, _, _, _, _, _, feature_dict = outputs

            umask_bool = umask.bool()
            labels_ = label[umask_bool]

            for mode in mode_names:
                feat_ = feature_dict[mode][umask_bool]
                for c in range(args.clsNum):
                    current_count = sum(x.size(0) for x in feats_by_mode_class[mode][c])
                    if current_count >= max_per_class:
                        continue
                    idx = (labels_ == c).nonzero(as_tuple=False).view(-1)
                    if idx.numel() == 0:
                        continue
                    remain = max_per_class - current_count
                    idx = idx[:remain]
                    feats_by_mode_class[mode][c].append(feat_[idx].detach().cpu())

    tsne_data = {}
    for mode in mode_names:
        all_x, all_y = [], []
        for c in range(args.clsNum):
            if len(feats_by_mode_class[mode][c]) == 0:
                continue
            x_c = torch.cat(feats_by_mode_class[mode][c], dim=0)
            y_c = torch.full((x_c.size(0),), c, dtype=torch.long)
            all_x.append(x_c)
            all_y.append(y_c)

        if len(all_x) == 0:
            raise RuntimeError(f'No features collected for mode={mode}')

        tsne_data[mode] = (torch.cat(all_x, dim=0).numpy(), torch.cat(all_y, dim=0).numpy())

    return tsne_data, class_names


def draw_tsne_figure(X, y, class_names, save_path, title='t-SNE'):
    n_samples = X.shape[0]
    perplexity = min(30, max(5, n_samples // 10))
    perplexity = min(perplexity, n_samples - 1)
    if perplexity < 2:
        raise RuntimeError(f'Not enough samples for TSNE: n_samples={n_samples}')

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=42,
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 7))
    for c, name in enumerate(class_names):
        idx = (y == c)
        if idx.sum() == 0:
            continue
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=12, alpha=0.75, label=name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[Saved] t-SNE figure: {save_path}')


def draw_tsne_figure_all(tsne_data, class_names, save_path, title='IEMOCAP t-SNE'):
    mode_order = ['text', 'audio', 'text_tl_audio', 'video', 'text_tl_video']
    pretty_names = {
        'text': '(a)text',
        'audio': '(b)audio',
        'text_tl_audio': '(c)text_tl_audio',
        'video': '(d)video',
        'text_tl_video': '(e)text_tl_video',
    }

    fig, axes = plt.subplots(1, 5, figsize=(20.48, 4.37))

    for ax, mode in zip(axes, mode_order):
        X, y = tsne_data[mode]
        n_samples = X.shape[0]
        perplexity = min(30, max(5, n_samples // 10))
        perplexity = min(perplexity, n_samples - 1)
        if perplexity < 2:
            raise RuntimeError(f'Not enough samples for TSNE in mode={mode}: n_samples={n_samples}')

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            random_state=42,
        )
        X_2d = tsne.fit_transform(X)

        for c, name in enumerate(class_names):
            idx = (y == c)
            if idx.sum() == 0:
                continue
            ax.scatter(X_2d[idx, 0], X_2d[idx, 1], s=10, alpha=0.75, label=name, linewidths=0)

        ax.set_title(pretty_names[mode], fontsize=20, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(0.965, 0.5),
        frameon=True,
        fontsize=12,
        markerscale=1.0,
        borderpad=0.6,
        handletextpad=0.6,
        labelspacing=0.4
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.93, 1.0])
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'[Saved] all-in-one t-SNE figure: {save_path}')

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
        text, _, video, audio, _, qmask, umask, label = [d.cuda() for d in data[:-1]]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # --- modality ablation (input-level): T-only / T+A / T+V ---
        
        if not getattr(args, 'use_audio', True):
            audio = torch.zeros_like(audio)
        if not getattr(args, 'use_video', True):
            video = torch.zeros_like(video)

        
        t_logit, a_logit, v_logit, t_hidden, a_hidden, v_hidden = model(text, audio, video, umask, qmask, lengths)
        


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
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_loss_a_kd = round(np.sum(losses_a_kd) / len(losses_a_kd), 4)
    avg_loss_v_kd = round(np.sum(losses_v_kd) / len(losses_v_kd), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, avg_loss_a_kd, avg_loss_v_kd


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

def model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, args, main_criterion, consistency_coef):
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    history = []

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc, _, _, _, train_fscore, train_loss_a_kd, train_loss_v_kd = train_or_eval_model(model, train_loader, epoch, optimizer, scheduler, True, main_criterion, consistency_coef)
        valid_loss, valid_acc, _, _, _, valid_fscore, valid_loss_a_kd, valid_loss_v_kd = train_or_eval_model(model, dev_loader, epoch, main_criterion=main_criterion, consistency_coef=consistency_coef)
        test_loss, test_acc, label, pred, _, test_fscore, test_loss_a_kd, test_loss_v_kd = train_or_eval_model(model, test_loader, epoch, main_criterion=main_criterion, consistency_coef=consistency_coef)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_fscore': test_fscore,
        })

        print(f'epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, train_fscore: {train_fscore} valid_loss: {valid_loss}, valid_acc: {valid_acc}, valid_fscore: {valid_fscore},test_loss: {test_loss}, test_acc: {test_acc}, test_fscore: {test_fscore}, time: {time.time()}')
        print(f'epoch: {epoch}, train_loss_a_kd: {train_loss_a_kd}, train_loss_v_kd: {train_loss_v_kd}, valid_loss_a_kd: {valid_loss_a_kd}, valid_loss_v_kd: {valid_loss_v_kd}, test_loss_a_kd: {test_loss_a_kd}, test_loss_v_kd: {test_loss_v_kd}')

        if best_fscore == None or test_fscore > best_fscore:
            prev_best = -1 if best_fscore is None else best_fscore
            best_fscore = test_fscore
            print(f'[NEW BEST] epoch={epoch}  test_fscore={best_fscore}  (prev_best={prev_best})')
            _SaveModel(model, './IEMOCAP/save_model', 'multimodal_fusion_best.bin')
            save_labels_and_preds(label, pred, f'IEMOCAP/save_model/multimodal_fusion_best.json')
            print(classification_report(label, pred, digits=4))
            print(f'done')

    print('\n=== Epoch Summary ===')
    print('epoch\ttrain_loss\ttrain_acc\tvalid_loss\tvalid_acc\ttest_loss\ttest_acc\ttest_fscore')
    for h in history:
        print(
            f"{h['epoch']}\t{h['train_loss']}\t{h['train_acc']}\t{h['valid_loss']}\t{h['valid_acc']}\t{h['test_loss']}\t{h['test_acc']}\t{h['test_fscore']}"
        )


# ===== 保存训练过程指标 =====
    save_dir = './IEMOCAP/save_model'
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(save_dir, f'multimodal_fusion_metrics_{timestamp}.csv')
    json_path = os.path.join(save_dir, f'multimodal_fusion_metrics_{timestamp}.json')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
           f,
         fieldnames=[
                'epoch',
                'train_loss', 'train_acc',
                'valid_loss', 'valid_acc',
                'test_loss', 'test_acc', 'test_fscore'
            ]
        )
        writer.writeheader()
        writer.writerows(history)

    with open(json_path, 'w', encoding='utf-8') as f:
     json.dump(history, f, indent=4, ensure_ascii=False)

    print(f'[Saved] metrics csv: {csv_path}')
    print(f'[Saved] metrics json: {json_path}')


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
    parser.add_argument('--clsNum', type=int, default=6, help='number of classes.')
    parser.add_argument('--train', type=str2bool, default=False, help='whether to train the model.')
    parser.add_argument('--loss_type', type=str, default='cb_focal', choices=['ce', 'focal', 'cb_focal', 'label_smoothing'], help='main classification loss type.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for focal loss.')
    parser.add_argument('--focal_alpha', type=float, default=None, help='scalar alpha for focal loss (overridden by class-balanced).')
    parser.add_argument('--cb_beta', type=float, default=0.9999, help='beta for class-balanced focal loss.')
    parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing factor for CE.')
    parser.add_argument('--consistency_coef', type=float, default=0.1, help='weight for symmetric KL consistency between modalities.')
    parser.add_argument('--kd_a_w', type=float, default=0.5, help='weight for audio KD loss (student=audio, teacher=text).')
    parser.add_argument('--kd_v_w', type=float, default=0.5, help='weight for video KD loss (student=video, teacher=text).')
    parser.add_argument('--use_audio', type=str2bool, default=True, help='whether to use audio modality input (audio).')
    parser.add_argument('--use_video', type=str2bool, default=True, help='whether to use video modality input (video tensor).')
    parser.add_argument('--draw_tsne', type=str2bool, default=False, help='whether to draw t-SNE instead of training.')
    parser.add_argument('--tsne_split', type=str, default='test', choices=['train', 'dev', 'test'], help='which split to visualize.')
    parser.add_argument('--tsne_all_in_one', type=str2bool, default=True,
                        help='draw text/audio/video/text_tl_audio/text_tl_video in one figure.')
    parser.add_argument('--tsne_max_per_class', type=int, default=200, help='max samples per class for t-SNE.')
    parser.add_argument('--tsne_save', type=str, default='./IEMOCAP/save_model/iemocap_tsne.png', help='path to save t-SNE figure.')
    args = parser.parse_args()

    # set seed
    seed_everything(args.seed)

    # create dataloader
    train_path = './feature/train_features.pkl'
    dev_path = './feature/dev_features.pkl'
    test_path = './feature/test_features.pkl'

    train_dataset = IEMOCAP_Dataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=7, collate_fn=train_dataset.collate_fn)

    dev_dataset = IEMOCAP_Dataset(dev_path)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=dev_dataset.collate_fn)

    test_dataset = IEMOCAP_Dataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=test_dataset.collate_fn)

    class_counts = compute_class_counts(train_dataset, args.clsNum)
    main_criterion = build_main_criterion(args, class_counts)

    # create model
    model = Transformer_Based_Model(args)
    # model = Fusion_model(args, 6)
    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    ckpt_path = './IEMOCAP/save_model/multimodal_fusion_best.bin'
    if (not args.train) or args.draw_tsne:
        model.load_state_dict(torch.load(ckpt_path))

    if args.draw_tsne:
        _, tsne_loader = get_tsne_loader(args, args.tsne_split)
        tsne_data, class_names = collect_tsne_features_all(
            model=model,
            data_loader=tsne_loader,
            args=args,
            max_per_class=args.tsne_max_per_class,
        )
        draw_tsne_figure_all(
            tsne_data,
            class_names,
            save_path=args.tsne_save,
            title=f'IEMOCAP'
        )
        raise SystemExit(0)

    num_training_steps = len(train_dataset) * args.epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, args, main_criterion, args.consistency_coef)


