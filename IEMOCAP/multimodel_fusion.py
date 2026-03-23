import numpy as np
import argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
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

from model import Transformer_Based_Model, MaskedKLDivLoss, MaskedNLLLoss, TestModel, Fusion_model
from dataset import *




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


IEMOCAP_LABELS = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
MELD_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

IEMOCAP_LABEL2ID = {name: i for i, name in enumerate(IEMOCAP_LABELS)}
HAP_CLASS_ID = IEMOCAP_LABEL2ID['hap']

IEMOCAP_IDENTITY_MAP = {
    0: IEMOCAP_LABEL2ID['ang'],
    1: IEMOCAP_LABEL2ID['exc'],
    2: IEMOCAP_LABEL2ID['fru'],
    3: IEMOCAP_LABEL2ID['hap'],
    4: IEMOCAP_LABEL2ID['neu'],
    5: IEMOCAP_LABEL2ID['sad'],
}

MELD_TO_IEMOCAP = {
    0: IEMOCAP_LABEL2ID['ang'],
    1: -1,
    2: IEMOCAP_LABEL2ID['fru'],
    3: IEMOCAP_LABEL2ID['hap'],
    4: IEMOCAP_LABEL2ID['neu'],
    5: IEMOCAP_LABEL2ID['sad'],
    6: -1,
}

IEMOCAP_EVAL_CLASS_IDS = list(range(len(IEMOCAP_LABELS)))
IEMOCAP_EVAL_LABEL_NAMES = IEMOCAP_LABELS


class MixedCorpusDataset(Dataset):
    def __init__(self, data_path, label_map=None):
        data = pk.load(open(data_path, 'rb'))
        self.text = data['text']
        self.audio = data['audio']
        self.video = data['video']
        self.audio_kd = data['audio_kd']
        self.video_kd = data['video_kd']
        self.speakers = data['speakers']
        self.labels = data['labels']
        self.vids = data['vids']
        self.dia2utt = data['dia2utt']
        self.label_map = label_map

        if self.label_map is not None:
            filtered_vids = []
            for vid in self.vids:
                mapped = [int(self.label_map[int(x)]) for x in self.labels[vid]]
                if any(x >= 0 for x in mapped):
                    filtered_vids.append(vid)
            self.vids = filtered_vids
        self.len = len(self.vids)

        self.padding_speaker_id = 2
        self.min_valid_speaker_id = 0
        self.max_valid_speaker_id = 1
        self.speaker_id_map = self._build_speaker_id_map()

    def _speaker_to_ids(self, speaker_seq):
        arr = np.array(speaker_seq)
        if arr.ndim == 0:
            return [int(arr.item())]
        if arr.ndim == 1:
            return [int(x) for x in arr.tolist()]
        return [int(x) for x in np.argmax(arr, axis=-1).tolist()]

    def _build_speaker_id_map(self):
        unique_ids = set()
        for vid in self.vids:
            ids = self._speaker_to_ids(self.speakers[vid])
            unique_ids.update(ids)

        sorted_ids = sorted(unique_ids)
        speaker_map = {}
        for raw_id in sorted_ids:
            speaker_map[raw_id] = int(raw_id) % 2
        return speaker_map

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        vid = self.vids[idx]
        if self.label_map is None:
            mapped_labels = [int(x) for x in self.labels[vid]]
            valid_indices = list(range(len(mapped_labels)))
        else:
            raw_mapped_labels = [int(self.label_map[int(x)]) for x in self.labels[vid]]
            valid_indices = [i for i, x in enumerate(raw_mapped_labels) if x >= 0]
            mapped_labels = [raw_mapped_labels[i] for i in valid_indices]
        speaker_ids = self._speaker_to_ids(self.speakers[vid])
        mapped_speakers = [self.speaker_id_map.get(int(speaker_ids[i]), self.min_valid_speaker_id) for i in valid_indices]
        text = np.array(self.text[vid])[valid_indices]
        audio = np.array(self.audio[vid])[valid_indices]
        video = np.array(self.video[vid])[valid_indices]
        audio_kd = np.array(self.audio_kd[vid])[valid_indices]
        video_kd = np.array(self.video_kd[vid])[valid_indices]
        return torch.FloatTensor(text),\
               torch.FloatTensor(audio),\
               torch.FloatTensor(video),\
               torch.FloatTensor(audio_kd),\
               torch.FloatTensor(video_kd),\
               torch.FloatTensor(np.array(mapped_speakers)),\
               torch.FloatTensor([1] * len(mapped_labels)),\
               torch.LongTensor(mapped_labels),\
               vid

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 5 else pad_sequence(dat[i], True) if i < 8 else dat[i].tolist() for i in dat]

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
    class_weight = torch.ones(args.clsNum)
    hap_loss_weight = getattr(args, 'hap_loss_weight', 1.0)
    if 0 <= HAP_CLASS_ID < args.clsNum and hap_loss_weight != 1.0:
        class_weight[HAP_CLASS_ID] = hap_loss_weight

    if args.loss_type == 'label_smoothing':
        return nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)

    if args.loss_type in ['focal', 'cb_focal']:
        alpha = None
        if args.loss_type == 'cb_focal' and class_counts is not None:
            alpha = compute_class_balanced_alpha(class_counts.float(), args.cb_beta)
            if 0 <= HAP_CLASS_ID < len(alpha) and hap_loss_weight != 1.0:
                alpha[HAP_CLASS_ID] = alpha[HAP_CLASS_ID] * hap_loss_weight
        elif args.focal_alpha is not None:
            alpha = torch.tensor([args.focal_alpha] * args.clsNum)
            if 0 <= HAP_CLASS_ID < len(alpha) and hap_loss_weight != 1.0:
                alpha[HAP_CLASS_ID] = alpha[HAP_CLASS_ID] * hap_loss_weight
        elif hap_loss_weight != 1.0:
            alpha = class_weight.clone()
        return FocalLoss(alpha=alpha, gamma=args.focal_gamma)

    return nn.CrossEntropyLoss(weight=class_weight)


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


def compute_class_counts_from_paths(path_and_label_map, cls_num):
    counts = torch.zeros(cls_num)
    for data_path, label_map in path_and_label_map:
        data = pk.load(open(data_path, 'rb'))
        for label_list in data['labels'].values():
            if label_map is None:
                mapped = torch.tensor([int(x) for x in label_list])
            else:
                mapped = torch.tensor([int(label_map[int(x)]) for x in label_list])
                mapped = mapped[mapped >= 0]
            if mapped.numel() > 0:
                counts += torch.bincount(mapped, minlength=cls_num)
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

def train_or_eval_model(model, data_loader, epoch, optimizer=None, scheduler=None, train=False, main_criterion=None, consistency_coef=0.0, class_indices=None, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
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
        qmask = torch.nan_to_num(qmask, nan=2.0, posinf=2.0, neginf=2.0)
        qmask = torch.round(qmask).clamp(min=0, max=2)

        qmask_min = int(qmask.min().item())
        qmask_max = int(qmask.max().item())
        if qmask_min < 0 or qmask_max > 2:
            raise RuntimeError(f'qmask index out of range before model forward: min={qmask_min}, max={qmask_max}')

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

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

        if class_indices is not None:
            class_idx_tensor = torch.tensor(class_indices, dtype=torch.long, device=logit_t.device)
            logit_t_task = torch.index_select(logit_t, 1, class_idx_tensor)
            logit_a_task = torch.index_select(logit_a, 1, class_idx_tensor)
            logit_v_task = torch.index_select(logit_v, 1, class_idx_tensor)

            global_to_local = torch.full((args.clsNum,), -1, dtype=torch.long, device=logit_t.device)
            global_to_local[class_idx_tensor] = torch.arange(len(class_indices), device=logit_t.device)
            labels_task = global_to_local[labels_]
        else:
            logit_t_task = logit_t
            logit_a_task = logit_a
            logit_v_task = logit_v
            labels_task = labels_

        if labels_task.numel() > 0:
            min_label = int(labels_task.min().item())
            max_label = int(labels_task.max().item())
            if min_label < 0 or max_label >= logit_t_task.size(1):
                raise RuntimeError(
                    f'Label index out of range: min={min_label}, max={max_label}, num_classes={logit_t_task.size(1)}, '
                    f'class_indices={class_indices}'
                )

        loss_kd_a = CE_Loss(args, logit_a_task, logit_t_task, hidden_a, hidden_t, labels_task)
        loss_kd_v = CE_Loss(args, logit_v_task, logit_t_task, hidden_v, hidden_t, labels_task)


        # loss_val = loss(logit_t , labels_) + 0.5 * (loss_kd_a + loss_kd_v)
        a = getattr(args, 'kd_a_w', 0.7)
        b = getattr(args, 'kd_v_w', 0.8)
        main_loss = main_criterion(logit_t_task, labels_task) if main_criterion is not None else base_ce(logit_t_task, labels_task)
        loss_val = main_loss + a * loss_kd_a + b * loss_kd_v

        if consistency_coef > 0:
            cons_loss = (
                symmetric_kl_divergence(logit_t_task, logit_a_task) +
                symmetric_kl_divergence(logit_t_task, logit_v_task) +
                symmetric_kl_divergence(logit_a_task, logit_v_task)
            )
            loss_val = loss_val + consistency_coef * cons_loss

        pred_ = torch.argmax(logit_t_task, dim=1)
        if class_indices is not None:
            pred_ = class_idx_tensor[pred_]
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


def save_confusion_matrix_plot(labels, preds, label_names, filename):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[WARN] skip confusion matrix plot ({filename}): {e}')
        return

    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

def model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loaders, args, main_criterion, consistency_coef):
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    history = []

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc, _, _, _, train_fscore, train_loss_a_kd, train_loss_v_kd = train_or_eval_model(model, train_loader, epoch, optimizer, scheduler, True, main_criterion, consistency_coef)
        valid_loss, valid_acc, _, _, _, valid_fscore, valid_loss_a_kd, valid_loss_v_kd = train_or_eval_model(model, dev_loader, epoch, main_criterion=main_criterion, consistency_coef=consistency_coef)
        test_metrics = {}
        for test_name, meta in test_loaders.items():
            test_loss, test_acc, label, pred, _, test_fscore, test_loss_a_kd, test_loss_v_kd = train_or_eval_model(
                model,
                meta['loader'],
                epoch,
                main_criterion=main_criterion,
                consistency_coef=consistency_coef,
                class_indices=meta['class_indices'],
            )
            
        mean_test_loss = float(np.mean([m['loss'] for m in test_metrics.values()]))
        mean_test_acc = float(np.mean([m['acc'] for m in test_metrics.values()]))
        mean_test_fscore = float(np.mean([m['fscore'] for m in test_metrics.values()]))
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'test_loss': mean_test_loss,
            'test_acc': mean_test_acc,
            'test_fscore': mean_test_fscore,
        })

        test_msg = ' | '.join([f"{n}: acc={m['acc']:.2f}, f1={m['fscore']:.2f}" for n, m in test_metrics.items()])
        print(f'epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, train_fscore: {train_fscore} valid_loss: {valid_loss}, valid_acc: {valid_acc}, valid_fscore: {valid_fscore}, test_avg_loss: {mean_test_loss}, test_avg_acc: {mean_test_acc}, test_avg_fscore: {mean_test_fscore}, time: {time.time()}')
        print(f'epoch: {epoch}, test_detail: {test_msg}')
        print(f'epoch: {epoch}, train_loss_a_kd: {train_loss_a_kd}, train_loss_v_kd: {train_loss_v_kd}, valid_loss_a_kd: {valid_loss_a_kd}, valid_loss_v_kd: {valid_loss_v_kd}')

        if best_fscore == None or mean_test_fscore > best_fscore:
            prev_best = -1 if best_fscore is None else best_fscore
            best_fscore = mean_test_fscore
            print(f'[NEW BEST] epoch={epoch}  avg_test_fscore={best_fscore}  (prev_best={prev_best})')
            _SaveModel(model, './IEMOCAP/save_model', 'multimodal_fusion_best.bin')
            for test_name, metric in test_metrics.items():
                save_labels_and_preds(metric['labels'], metric['preds'], f'IEMOCAP/save_model/multimodal_fusion_best_{test_name}.json')
                global_to_local = {gid: i for i, gid in enumerate(metric['class_indices'])}
                rel_labels = np.array([global_to_local[int(x)] for x in metric['labels']])
                rel_preds = np.array([global_to_local[int(x)] for x in metric['preds']])
                save_confusion_matrix_plot(
                    rel_labels,
                    rel_preds,
                    metric['label_names'],
                    f'IEMOCAP/save_model/multimodal_fusion_best_{test_name}_cm.png'
                )
                print(f'[BEST][{test_name}]')
                print(classification_report(rel_labels, rel_preds, digits=4, target_names=metric['label_names'], zero_division=0))
            print(f'done')

    print('\n=== Epoch Summary ===')
    print('epoch\ttrain_loss\ttrain_acc\tvalid_loss\tvalid_acc\ttest_loss\ttest_acc\ttest_fscore')
    for h in history:
        print(
            f"{h['epoch']}\t{h['train_loss']}\t{h['train_acc']}\t{h['valid_loss']}\t{h['valid_acc']}\t{h['test_loss']}\t{h['test_acc']}\t{h['test_fscore']}"
        )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training.')
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 regularization weight.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training.')
    parser.add_argument('--seed', type=int, default=42, help='random seed for training.')
    parser.add_argument('--epochs', type=int, default=25, help='epoch for training.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden dimension.')
    parser.add_argument('--n_head', type=int, default=8, help='number of heads.')
    parser.add_argument('--num_layers', type=int, default=6, help='number of heads.')
    parser.add_argument('--n_rounds', type=int, default=1, help='number of interaction rounds for Transformer Model.')
    parser.add_argument('--temp', type=float, default=2.0, help='temperature for contrastive learning.')
    parser.add_argument('--clsNum', type=int, default=len(IEMOCAP_LABELS), help='number of classes for IEMOCAP label space.')
    parser.add_argument('--train', type=str2bool, default=True, help='whether to train the model.')
    parser.add_argument('--loss_type', type=str, default='cb_focal', choices=['ce', 'focal', 'cb_focal', 'label_smoothing'], help='main classification loss type.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for focal loss.')
    parser.add_argument('--focal_alpha', type=float, default=None, help='scalar alpha for focal loss (overridden by class-balanced).')
    parser.add_argument('--cb_beta', type=float, default=0.9999, help='beta for class-balanced focal loss.')
    parser.add_argument('--hap_loss_weight', type=float, default=1.5, help='extra loss weight multiplier for hap class.')
    parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing factor for CE.')
    parser.add_argument('--consistency_coef', type=float, default=0.1, help='weight for symmetric KL consistency between modalities.')
    parser.add_argument('--kd_a_w', type=float, default=0.5, help='weight for audio KD loss (student=audio, teacher=text).')
    parser.add_argument('--kd_v_w', type=float, default=0.5, help='weight for video KD loss (student=video, teacher=text).')
    parser.add_argument('--iemocap_train_path', type=str, default='./feature/train_features.pkl', help='IEMOCAP train pkl path.')
    parser.add_argument('--iemocap_dev_path', type=str, default='./feature/dev_features.pkl', help='IEMOCAP dev pkl path.')
    parser.add_argument('--iemocap_test_path', type=str, default='./feature/test_features.pkl', help='IEMOCAP test pkl path.')
    parser.add_argument('--meld_train_path', type=str, default='../MELD/feature/train_features.pkl', help='MELD train pkl path.')
    args = parser.parse_args()

    # set seed
    seed_everything(args.seed)

    
    iemocap_train_dataset = MixedCorpusDataset(args.iemocap_train_path, label_map=IEMOCAP_IDENTITY_MAP)
    meld_train_dataset = MixedCorpusDataset(args.meld_train_path, label_map=MELD_TO_IEMOCAP)
    train_dataset = ConcatDataset([iemocap_train_dataset, meld_train_dataset])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=7, collate_fn=iemocap_train_dataset.collate_fn)

    dev_dataset = MixedCorpusDataset(args.iemocap_dev_path, label_map=IEMOCAP_IDENTITY_MAP)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=dev_dataset.collate_fn)

    iemocap_test_dataset = MixedCorpusDataset(args.iemocap_test_path, label_map=IEMOCAP_IDENTITY_MAP)
    test_loaders = {
        'iemocap': {
            'loader': DataLoader(iemocap_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, collate_fn=iemocap_test_dataset.collate_fn),
            'class_indices': IEMOCAP_EVAL_CLASS_IDS,
            'label_names': IEMOCAP_EVAL_LABEL_NAMES,
        }
    }

    class_counts = compute_class_counts_from_paths([
        (args.iemocap_train_path, IEMOCAP_IDENTITY_MAP),
        (args.meld_train_path, MELD_TO_IEMOCAP),
    ], args.clsNum)
    main_criterion = build_main_criterion(args, class_counts)

    # create model
    model = Transformer_Based_Model(args)
    # model = Fusion_model(args, 6)
    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if not args.train:
        model.load_state_dict(torch.load('./IEMOCAP/save_model/multimodal_fusion_best.bin'))

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loaders, args, main_criterion, args.consistency_coef)


