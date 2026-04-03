import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle
import os


# class IEMOCAPDataset(Dataset):
#     def __init__(self, data):
#         self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
#         self.session_dataset = data

#     def __len__(self):
#         return len(self.session_dataset)

#     def __getitem__(self, idx):
#         return self.session_dataset[idx]


class AudioDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.dialogue_ids = self.data['dialogue_id']
        self.utterance_ids = self.data['utterance_id']
        self.features = self.data['features']
        self.labels = self.data['label']
        self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        self.label_to_index = {label: idx for idx, label in enumerate(self.emoList)}

    def load_data(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        label_index = self.label_to_index[label]
        return [feature, label_index]


class MOSEI_Dataset(Dataset):
    def __init__(self, data_path):
        self.emoList = [
            'strongly_negative',
            'negative',
            'weakly_negative',
            'neutral',
            'weakly_positive',
            'positive',
            'strongly_positive',
        ]
        self.label_to_index = {label: idx for idx, label in enumerate(self.emoList)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.emoList)}

        data_path = self._resolve_data_path(data_path)
        data = pickle.load(open(data_path, 'rb'))
        self.text = data['text']
        self.audio = data['audio']
        self.video = data['video']
        self.speakers = data['speakers']
        self.labels = data['labels']
        self.vids = data['vids']
        self.dia2utt = data['dia2utt']
        self.len = len(self.vids)

    @staticmethod
    def _resolve_data_path(data_path):
        if os.path.exists(data_path):
            return data_path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        normalized = data_path.lstrip('./').lstrip('/')
        candidate = os.path.join(base_dir, normalized)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f'Feature file not found: {data_path}')

    @staticmethod
    def _normalize_modality(x):
        arr = np.asarray(x)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        elif arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 0:
            arr = arr.reshape(1, 1)
        return arr.astype(np.float32)

    @staticmethod
    def _normalize_labels(labels):
        arr = np.asarray(labels).reshape(-1).astype(np.int64)
        if arr.size == 0:
            return arr
        if arr.min() < 0 and arr.max() <= 3:
            arr = arr + 3
        arr = np.clip(arr, 0, 6)
        return arr

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        vid = self.vids[idx]
        text = self._normalize_modality(self.text[vid])
        audio = self._normalize_modality(self.audio[vid])
        video = self._normalize_modality(self.video[vid])

        speakers = np.asarray(self.speakers[vid], dtype=np.float32)
        if speakers.ndim == 0:
            speakers = speakers.reshape(1)

        labels = self._normalize_labels(self.labels[vid])

        return (
            torch.FloatTensor(text),
            torch.FloatTensor(audio),
            torch.FloatTensor(video),
            torch.FloatTensor(speakers),
            torch.FloatTensor([1] * len(labels)),
            torch.LongTensor(labels),
            vid,
        )

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [
            pad_sequence(dat[i]) if i < 5 else pad_sequence(dat[i], True) if i < 8 else dat[i].tolist()
            for i in dat
        ]

