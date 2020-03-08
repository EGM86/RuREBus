import os
from os.path import join, exists
from typing import List, Union

from .datasets import NerDataset, BertNerDataset, RelDataset, BertRelDataset, RBertDataset
from .span_utils import NER_TAG_REGEX, tags_to_spans


class NerEvaluator:
    """
    Evaluator for NER
    """
    def save_result(self, dataset: NerDataset, result: List[str], results_folder: str):
        if not exists(results_folder):
            os.mkdir(results_folder)

        current_doc = None
        current_entity = 1
        for i in range(len(dataset)):
            doc, sentence = dataset.sentences[i]
            if current_doc is None or current_doc != doc:
                current_doc = doc
                current_entity = 1
            with open(join(results_folder, current_doc + '.ann'), 'a', encoding='utf-8') as f:
                labels = result[i]
                spans = tags_to_spans(labels, NER_TAG_REGEX)
                for span in spans:
                    entity_type = span[0]
                    start_token = sentence.tokens[span[1][0]]
                    end_token = sentence.tokens[span[1][1]]
                    start = start_token.offset
                    end = end_token.offset + len(end_token.word)
                    f.write('T%d\t%s %d %d\n' % (current_entity, entity_type, start, end))
                    current_entity += 1


class RelEvaluator:
    """
    Common evaluator for Relationship Extraction from all models
    """
    def save_result(self, dataset: Union[RelDataset, RBertDataset, BertRelDataset],
                    result: List[str], results_folder: str):
        if not exists(results_folder):
            os.mkdir(results_folder)

        current_doc = None
        current_relation = 1
        for i in range(len(dataset)):
            doc, sample = dataset.samples[i]
            if current_doc is None or current_doc != doc:
                current_doc = doc
                current_relation = 1
            with open(join(results_folder, current_doc + '.ann'), 'a', encoding='utf-8') as f:
                label = result[i]
                if label != 'OUT':
                    f.write('R%d\t%s Arg1:T%d Arg2:T%d\n' % (current_relation, label, sample.e1.idx, sample.e2.idx))
                    current_relation += 1


class BertNerEvaluator:
    """
    Evaluator for NER using BERT results
    """
    def save_result(self, dataset: BertNerDataset, result: List[List[str]], results_folder: str):
        if not exists(results_folder):
            os.mkdir(results_folder)

        current_doc = None
        current_entity = 1
        for i in range(len(dataset)):
            doc, sentence = dataset.sentences[i]
            if current_doc is None or current_doc != doc:
                current_doc = doc
                current_entity = 1
            with open(join(results_folder, current_doc + '.ann'), 'a', encoding='utf-8') as f:
                labels = result[i]
                spans = tags_to_spans(labels, NER_TAG_REGEX)
                for span in spans:
                    entity_type = span[0]
                    start = sentence.tokens.offsets[span[1][0]][0] + sentence.offset
                    end = sentence.tokens.offsets[span[1][1]][1] + sentence.offset
                    f.write('T%d\t%s %d %d\n' % (current_entity, entity_type, start, end))
                    current_entity += 1
