from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .torch_utils import pad_matrix
from .indexers import DocumentIndexer, BertIndexer
from .fields import Sentence, Document, BertSentence, RelationSample
from .samplers import RelationsSampler


class NerDataset(Dataset):
    def __init__(self,
                 indexer: DocumentIndexer,
                 documents: List[Document],
                 device: torch.device):
        self.indexer = indexer
        self.sentences = []  # type: List[Tuple[str, Sentence]]
        self.device = device

        for document in documents:
            for sentence in document.sentences:
                self.sentences.append((document.file_name, sentence))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item: int):
        indices = self.indexer.sentence_to_indexes(self.sentences[item][1])

        return torch.tensor(indices['words']), \
               pad_sequence([torch.tensor(word_chars) for word_chars in indices['chars']], batch_first=True), \
               torch.tensor(indices['cases']), \
               torch.tensor(indices['pos']), \
               torch.tensor(indices['ner'])

    def collate_fn(self, batch):
        words, chars, cases, pos, ner = list(zip(*batch))

        words = pad_sequence(words, batch_first=True).to(self.device)
        chars = pad_matrix(chars).to(self.device)
        cases = pad_sequence(cases, batch_first=True).to(self.device)
        pos = pad_sequence(pos, batch_first=True).to(self.device)
        ner = pad_sequence(ner, batch_first=True).to(self.device)

        return words, chars, cases, pos, ner


class RelDataset(Dataset):
    def __init__(self,
                 label_indexer: DocumentIndexer,
                 sampler: RelationsSampler,
                 documents: List[Document],
                 device: torch.device):
        self.indexer = label_indexer
        self.samples = []  # type: List[Tuple[str, RelationSample]]
        self.device = device
        self.sampler = sampler

        for document in documents:
            document_samples = self.sampler.sample_document(document)
            for sample in document_samples:
                self.samples.append((document.file_name, sample))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int):
        indices = self.indexer.sample_to_indexes(self.samples[item][1])

        return torch.tensor(indices['words']), \
               pad_sequence([torch.tensor(word_chars) for word_chars in indices['chars']], batch_first=True), \
               torch.tensor(indices['cases']), \
               torch.tensor(indices['pos']), \
               torch.tensor(indices['ner']), \
               torch.tensor(indices['relation'])

    def collate_fn(self, batch):
        words, chars, cases, pos, ner, labels = list(zip(*batch))

        words = pad_sequence(words, batch_first=True).to(self.device)
        chars = pad_matrix(chars).to(self.device)
        cases = pad_sequence(cases, batch_first=True).to(self.device)
        pos = pad_sequence(pos, batch_first=True).to(self.device)
        ner = pad_sequence(ner, batch_first=True).to(self.device)
        labels = torch.stack(labels).to(self.device)
        return words, chars, cases, pos, ner, labels


class BertNerDataset(Dataset):

    def __init__(self,
                 label_indexer: BertIndexer,
                 documents: List[Document],
                 device: torch.device):
        self.indexer = label_indexer
        self.sentences = []  # type: List[Tuple[str, BertSentence]]
        self.device = device

        for document in documents:
            for sentence in document.sentences:
                self.sentences.append((document.file_name, sentence))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item: int):
        sentence = self.sentences[item][1]
        labels = self.indexer.get_ner_labels(sentence)
        tokens = sentence.tokens.ids
        return torch.tensor(tokens), torch.tensor(labels)

    def collate_fn(self, batch):
        tokens, labels = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True).to(self.device)
        labels = pad_sequence(labels, batch_first=True).to(self.device)

        return tokens, labels


class BertRelDataset(Dataset):
    def __init__(self,
                 indexer: BertIndexer,
                 sampler: RelationsSampler,
                 documents: List[Document],
                 device: torch.device):
        self.indexer = indexer
        self.samples = []  # type: List[Tuple[str, RelationSample]]
        self.device = device
        self.sampler = sampler

        for document in documents:
            document_samples = self.sampler.sample_document(document)
            for sample in document_samples:
                self.samples.append((document.file_name, sample))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int):
        sample = self.samples[item][1]
        ner, rel = self.indexer.get_rel_labels(sample)
        tokens = sample.tokens.ids
        return torch.tensor(tokens), torch.tensor(ner), torch.tensor(rel)

    def collate_fn(self, batch):
        tokens, ner, rel = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True).to(self.device)
        ner = pad_sequence(ner, batch_first=True).to(self.device)
        rel = torch.stack(rel).to(self.device)

        return tokens, ner, rel


class RBertDataset(Dataset):
    def __init__(self,
                 indexer: BertIndexer,
                 sampler: RelationsSampler,
                 documents: List[Document],
                 device: torch.device):
        self.indexer = indexer
        self.samples = []  # type: List[Tuple[str, RelationSample]]
        self.device = device
        self.sampler = sampler

        for document in documents:
            document_samples = self.sampler.sample_document(document)
            for sample in document_samples:
                self.samples.append((document.file_name, sample))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int):
        sample = self.samples[item][1]
        e1 = sample.e1 if sample.e1.start < sample.e2.start else sample.e2
        e2 = sample.e1 if sample.e1.start >= sample.e2.start else sample.e2
        _, rel = self.indexer.get_rel_labels(sample)

        tokens = sample.tokens.ids
        tokens.insert(e1.start, 109)
        tokens.insert(e1.end + 2, 109)
        tokens.insert(e2.start + 2, 108)
        tokens.insert(e2.end + 4, 108)

        return torch.tensor(tokens), (e1.start+1, e1.end+1), (e2.start+3, e2.end+3), torch.tensor(rel)

    def collate_fn(self, batch):
        tokens, e1_pos, e2_pos,  rel = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True).to(self.device)
        rel = torch.stack(rel).to(self.device)
        return tokens, e1_pos, e2_pos, rel
