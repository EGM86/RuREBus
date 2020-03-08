from typing import List

import torch
from allennlp.nn.util import sequence_cross_entropy_with_logits
from torch import nn

from .vocab import Vocab


class NerTagger:
    """
    Helper for using different models for Sequence Labeling
    """
    def __init__(self, label_vocab: Vocab, model: nn.Module):
        self.vocab = label_vocab
        self.model = model

    def forward(self, *args):
        args = list(args)
        words = args[0]
        labels = args[-1]
        features = args[:-1]
        mask = (words != 0).to(torch.long)
        logits = self.model(*features)
        loss = None
        if labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        return logits, mask, loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def decode(self, logits, mask):
        pred = torch.argmax(logits, dim=2).tolist()  # batch, seq_len
        return self.decode_labels(pred, mask)

    def decode_labels(self, labels, mask):
        lengths = torch.sum(mask, dim=1).tolist()  # batch
        idx2label = self.vocab.idx2label

        all_tags = []
        for (tag_idxs, length) in zip(labels, lengths):
            tags = []
            for idx, _ in zip(tag_idxs, range(length)):
                tags.append(idx2label[idx])
            all_tags.append(tags)
        return all_tags

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()


class RelClassifier:
    """
    Helper for using different models for Sequence Classification
    """
    def __init__(self, label_vocab: Vocab, model: nn.Module):
        self.vocab = label_vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args):
        args = list(args)
        labels = args[-1]
        features = args[:-1]
        logits = self.model(*features)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return logits, loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def decode(self, logits):
        _, pred = logits.topk(1)  # batch, 1
        pred = pred.squeeze(dim=-1).tolist()
        return self.decode_labels(pred)

    def decode_labels(self, labels: List[int]) -> List[str]:
        idx2label = self.vocab.idx2label
        return [idx2label[idx] for idx in labels]

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()
