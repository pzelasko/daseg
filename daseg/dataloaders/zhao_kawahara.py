from typing import Dict

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from daseg import DialogActCorpus, Call

TEXT_PAD_ID = 0
LABEL_PAD_ID = nn.CrossEntropyLoss().ignore_index


class ZhaoKawaharaDataset(Dataset):
    def __init__(self, corpus: DialogActCorpus, word2idx: Dict[str, int], tag2idx: Dict[str, int]):
        super().__init__()
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.turns = list(map(Call, corpus.turns))
        x = 1

    def __getitem__(self, item):
        turn = self.turns[item]
        encoded_segments = turn.encode(use_joint_coding=True, continuations_allowed=False, add_turn_token=False)
        token_indices = [self.word2idx[word] for segment in encoded_segments for word in segment.words]
        act_indices = [self.tag2idx[act] for segment in encoded_segments for act in segment.encoded_acts]
        return token_indices, act_indices

    def __len__(self):
        return len(self.turns)


def padding_collate_fn(batch):
    batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)  # sort by length
    lengths = [len(sample[0]) for sample in batch]
    max_len = lengths[0]
    text = torch.tensor([
        entry[0] + [TEXT_PAD_ID] * (max_len - length)
        for entry, length in zip(batch, lengths)
    ], dtype=torch.int64)
    label = torch.tensor([
        entry[1] + [LABEL_PAD_ID] * (max_len - length)
        for entry, length in zip(batch, lengths)
    ], dtype=torch.int64)
    return text, lengths, label
