from typing import List


class Vocab:
    """
    A common representation for index vocabulary: str -> int
    """
    def __init__(self, lowercase=False, paddings=True):
        self.label2idx = {}
        self.idx2label = {}
        if paddings:
            self.label2idx['<pad>'] = 0
            self.idx2label[0] = '<pad>'
        self.lowercase = lowercase

    def fill(self, labels: List[str]):
        for label in labels:
            self.add(label)

    def add(self, label: str):
        if self.lowercase:
            label = label.lower()
        if label not in self.label2idx:
            self.label2idx[label] = len(self.label2idx)
            self.idx2label[len(self.idx2label)] = label

    def __getitem__(self, item: str):
        if self.lowercase:
            item = item.lower()
        return self.label2idx[item]

    def __len__(self):
        return len(self.label2idx)

    def __contains__(self, item: str):
        return item in self.label2idx
