from collections import defaultdict
from os.path import join

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import f1_span_score, f1_score
from .span_utils import NER_TAG_REGEX

from .modules import NerTagger, RelClassifier


class MockWriter:
    def __init__(self):
        pass

    def add_scalar(self, name, value, index):
        pass


class NerTrainer:
    def __init__(self, module: NerTagger, train_iterator, test_iterator, writer: SummaryWriter,
                 save_folder: str, lr=2e-5):
        self.module = module
        self.optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.best_score = 0
        self.writer = writer if writer is not None else MockWriter()
        self.epoch = -1
        self.counters = None
        self.save_folder = save_folder
        self.reset_counters()

    def reset_counters(self):
        self.counters = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int)
        }

    def calculate_metrics(self):
        tp, fp, fn = 0, 0, 0
        for val in self.counters['true_positives'].values():
            tp += val
        for val in self.counters['false_positives'].values():
            fp += val
        for val in self.counters['false_negatives'].values():
            fn += val
        precison = tp / (fp + tp) if fp + tp != 0 else 0
        recall = tp / (tp + fn) if fn + tp != 0 else 0
        f1_score = 2*precison*recall / (precison + recall) if precison + recall != 0 else 0
        return precison, recall, f1_score

    def train_epoch(self):
        self.epoch += 1
        self.reset_counters()
        total_loss = 0
        self.module.train()
        precision, recall, f1_score = 0, 0, 0
        for batch_idx, (*features, labels) in enumerate(self.train_iterator):
            self.optimizer.zero_grad()
            logits, mask, loss = self.module(*features, labels)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            self.evaluate(logits, mask, labels)
            precision, recall, f1_score = self.calculate_metrics()
            print('\rLoss: %4f, Precision: %4f, Recall: %4f, F1_score: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), precision, recall, f1_score,
                batch_idx + 1, len(self.train_iterator)), end='')
        print()
        self.writer.add_scalar('precision/train', precision, self.epoch)
        self.writer.add_scalar('recall/train', recall, self.epoch)
        self.writer.add_scalar('f1_marco/train', f1_score, self.epoch)
        self.writer.add_scalar('loss/train', total_loss, self.epoch)

    def test_epoch(self):
        self.reset_counters()
        total_loss = 0
        self.module.eval()
        precision, recall, f1_score = 0, 0, 0
        for batch_idx, (*features, labels) in enumerate(self.test_iterator):
            logits, mask, loss = self.module(*features, labels)
            total_loss += loss.item()
            self.evaluate(logits, mask, labels)
            precision, recall, f1_score = self.calculate_metrics()
            print('\rLoss: %4f, Precision: %4f, Recall: %4f, F1_score: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), precision, recall, f1_score,
                batch_idx + 1, len(self.test_iterator)), end='')
        print()
        if f1_score >= self.best_score:
            print('New best score: ', f1_score)
            self.best_score = f1_score
            # save model
            torch.save(self.module.model.state_dict(), join(self.save_folder, 'best_ner.pt'))

        self.writer.add_scalar('precision/test', precision, self.epoch)
        self.writer.add_scalar('recall/test', recall, self.epoch)
        self.writer.add_scalar('f1_marco/test', f1_score, self.epoch)
        self.writer.add_scalar('loss/test', total_loss, self.epoch)

    def evaluate(self, logits, mask, labels):
        pred_tags = self.module.decode(logits, mask)
        gold_tags = self.module.decode_labels(labels.tolist(), mask)
        for pred, gold in zip(pred_tags, gold_tags):
            tp, fp, fn = f1_span_score(pred, gold, NER_TAG_REGEX)
            for ent, val in tp.items():
                self.counters['true_positives'][ent] += val
            for ent, val in fp.items():
                self.counters['false_positives'][ent] += val
            for ent, val in fn.items():
                self.counters['false_negatives'][ent] += val

    def decode(self, iterator):
        self.module.eval()
        result = []
        for batch_idx, (*features, labels) in tqdm(enumerate(iterator)):
            logits, mask, _ = self.module(*features, labels)
            pred_tags = self.module.decode(logits, mask)
            result.extend(pred_tags)
        return result


class RelTrainer:
    def __init__(self, module: RelClassifier, writer: SummaryWriter,
                 save_folder: str, lr=2e-5):
        self.module = module
        self.optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        self.best_score = 0
        self.writer = writer if writer is not None else MockWriter()
        self.epoch = -1
        self.counters = None
        self.save_folder = save_folder
        self.reset_counters()

    def reset_counters(self):
        self.counters = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int)
        }

    def calculate_metrics(self):
        tp, fp, fn = 0, 0, 0
        for label, val in self.counters['true_positives'].items():
            if label != 'OUT':
                tp += val
        for label, val in self.counters['false_positives'].items():
            if label != 'OUT':
                fp += val
        for label, val in self.counters['false_negatives'].items():
            if label != 'OUT':
                fn += val
        precison = tp / (fp + tp) if fp + tp != 0 else 0
        recall = tp / (tp + fn) if fn + tp != 0 else 0
        f1_score = 2 * precison * recall / (precison + recall) if precison + recall != 0 else 0
        return precison, recall, f1_score

    def train_epoch(self, iterator):
        self.epoch += 1
        self.reset_counters()
        total_loss = 0
        self.module.train()
        precision, recall, f1_score = 0, 0, 0
        for batch_idx, (*features, labels) in enumerate(iterator):
            self.optimizer.zero_grad()
            logits, loss = self.module(*features, labels)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            self.evaluate(logits, labels)
            precision, recall, f1_score = self.calculate_metrics()
            print('\rLoss: %4f, Precision: %4f, Recall: %4f, F1_score: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), precision, recall, f1_score,
                batch_idx + 1, len(iterator)), end='')
        print()
        self.writer.add_scalar('precision/train', precision, self.epoch)
        self.writer.add_scalar('recall/train', recall, self.epoch)
        self.writer.add_scalar('f1_marco/train', f1_score, self.epoch)
        self.writer.add_scalar('loss/train', total_loss / len(iterator), self.epoch)

        torch.save(self.module.model.state_dict(), join(self.save_folder, 'epoch_%d.pt' % self.epoch))

    def test_epoch(self, iterator):
        self.reset_counters()
        total_loss = 0
        self.module.eval()
        precision, recall, f1_score = 0, 0, 0
        for batch_idx, (*features, labels) in enumerate(iterator):
            logits, loss = self.module(*features, labels)
            total_loss += loss.item()
            self.evaluate(logits, labels)
            precision, recall, f1_score = self.calculate_metrics()
            print('\rLoss: %4f, Precision: %4f, Recall: %4f, F1_score: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), precision, recall, f1_score,
                batch_idx + 1, len(iterator)), end='')
        print()
        if f1_score >= self.best_score:
            print('New best score: ', f1_score)
            self.best_score = f1_score
            # save model
            torch.save(self.module.model.state_dict(), join(self.save_folder, 'best.pt'))

        self.writer.add_scalar('precision/test', precision, self.epoch)
        self.writer.add_scalar('recall/test', recall, self.epoch)
        self.writer.add_scalar('f1_marco/test', f1_score, self.epoch)
        self.writer.add_scalar('loss/test', total_loss / len(iterator), self.epoch)

    def evaluate(self, logits, labels):
        pred = self.module.decode(logits)
        gold = self.module.decode_labels(labels.tolist())

        tp, fp, fn = f1_score(pred, gold)
        for ent, val in tp.items():
                self.counters['true_positives'][ent] += val
        for ent, val in fp.items():
                self.counters['false_positives'][ent] += val
        for ent, val in fn.items():
                self.counters['false_negatives'][ent] += val

    def decode(self, iterator):
        self.module.eval()
        result = []

        print('Batches total: ', len(iterator))
        for batch_idx, (*features, labels) in tqdm(enumerate(iterator)):
            logits, _ = self.module(*features, labels)
            preds = self.module.decode(logits)
            result.extend(preds)
        return result
