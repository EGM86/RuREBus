from typing import List, Tuple

import pymorphy2
from tokenizers import BertWordPieceTokenizer

from rurebus.fields import Document, Sentence, Token, Entity, Relation, NerAnnotation, RelAnnotation, BertSentence
from os import listdir
from os.path import join, splitext, exists
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize


class RuREBusReader:
    def __init__(self, debug_file):
        self.debug_file = debug_file
        self.current_document = None

    def read_annotations(self, annotations_file: str) -> Tuple[List[NerAnnotation], List[RelAnnotation]]:
        ner_annotations = []  # typing: List[NerAnnotation]
        rel_annotations = []  # typing: List[RelAnnotation]

        with open(annotations_file, encoding='utf-8') as f:
            for line in f:
                if line.startswith('T'):
                    try:
                        entity_idx, entity, *value = line.strip().split('\t')
                    except Exception as e:
                        print(annotations_file, line, e)
                        exit(1)
                    entity_type, entity_start, entity_end = entity.split()
                    entity_start, entity_end = int(entity_start), int(entity_end)
                    entity_idx = int(entity_idx[1:])
                    annotation = NerAnnotation(
                        idx=entity_idx, type=entity_type, start=entity_start, end=entity_end, value=value
                    )
                    ner_annotations.append(annotation)
                elif line.startswith('R'):
                    relation_idx, relation = line.strip().split('\t')
                    relation_type, arg1, arg2 = relation.split()
                    relation_idx, arg1, arg2 = int(relation_idx[1:]), int(arg1[6:]), int(arg2[6:])
                    annotation = RelAnnotation(
                        idx=relation_idx, type=relation_type, arg1=arg1, arg2=arg2
                    )
                    rel_annotations.append(annotation)
        return ner_annotations, rel_annotations

    def process_relations(self, entities: List[Entity], rel_annotations: List[RelAnnotation]) -> List[Relation]:
        result = []
        entities_indices = set([entity.idx for entity in entities])
        for relation in rel_annotations:
            if relation.arg1 in entities_indices or relation.arg2 in entities_indices:
                # both entities should be in one paragraph
                if not (relation.arg1 in entities_indices and relation.arg2 in entities_indices):
                    with open(self.debug_file, 'a', encoding='utf-8') as f:
                        f.write("%s:%s\n" % (self.current_document.file_name, str(relation)))
                    continue

                arg1pos, arg2pos = 0, 0
                for i, entity in enumerate(entities):
                    if entity.idx == relation.arg1:
                        arg1pos = i
                    if entity.idx == relation.arg2:
                        arg2pos = i
                rel = Relation(idx=relation.idx, type=relation.type, arg1=arg1pos, arg2=arg2pos)
                result.append(rel)
        return result

    def find_entity(self, index: int, ner_annotations: List[NerAnnotation]):
        for annotation in ner_annotations:
            if annotation.end > index >= annotation.start:
                return annotation
        return None

    def read_document(self, file_name: str, folder_path: str) -> Document:
        raise NotImplemented()

    def read_folder(self, folder_path: str) -> List[Document]:
        raise NotImplemented


class SimpleReader(RuREBusReader):
    """
    Reader for non-BERT models
    """
    def __init__(self, debug_file: str):
        super().__init__(debug_file)
        self.current_document = None  # type: Document
        self.debug_file = debug_file
        self.morph_analyzer = pymorphy2.MorphAnalyzer()

    def read_tokens(self, text_file: str) -> List[List[Token]]:
        result = []
        with open(text_file, encoding='utf-8') as f:
            data = f.read()
            sentences = sent_tokenize(data, language='russian')
            last_index = 0
            for sentence in sentences:
                words = word_tokenize(sentence, language='russian')
                tokens = []  # type: List[Token]
                for word in words:
                    if word == "``" or word == "''":
                        word = '"'
                    i = data.find(word, last_index)
                    pos_tag = self.morph_analyzer.parse(word)[0].tag.POS
                    if pos_tag is None:
                        pos_tag = 'PUNCT'
                    tokens.append(Token(offset=i, word=word,
                                        pos_tag=pos_tag))
                    last_index = i + len(word) - 1
                result.append(tokens)
        return result

    def generate_sentences(self, tokens_by_sentence: List[List[Token]], ner_annotations: List[NerAnnotation],
                           rel_annotations: List[RelAnnotation]) -> List[Sentence]:
        result = []
        for sentence_tokens in tokens_by_sentence:
            current_entity = None
            sentence = Sentence()
            sentence.tokens = sentence_tokens
            for i, token in enumerate(sentence_tokens):
                ann = self.find_entity(token.offset, ner_annotations)
                if ann is not None:
                    if current_entity is None:
                        # found new entity
                        current_entity = Entity(
                            idx=ann.idx, label=ann.type, start=i, end=i
                        )
                    else:
                        if current_entity.idx == ann.idx:
                            # shift end of current entity
                            current_entity.end = i
                        else:
                            # consecutive entities
                            sentence.entities.append(current_entity)
                            current_entity = Entity(
                                idx=ann.idx, label=ann.type, start=i, end=i
                            )
                else:
                    if current_entity is not None:
                        sentence.entities.append(current_entity)
                        current_entity = None
            if current_entity is not None:
                sentence.entities.append(current_entity)
            sentence.relations = self.process_relations(sentence.entities, rel_annotations)
            result.append(sentence)
        return result

    def generate_test_sentences(self, tokens_by_sentence: List[List[Token]]):
        result = []
        for sentence_tokens in tokens_by_sentence:
            sentence = Sentence()
            sentence.tokens = sentence_tokens
            result.append(sentence)
        return result

    def read_document(self, file_name: str, folder_path: str) -> Document:
        document = Document(file_name)
        # save document for debugging
        self.current_document = document
        text_file = join(folder_path, file_name + '.txt')
        annotations_file = join(folder_path, file_name + '.ann')

        tokens_by_sentence = self.read_tokens(text_file)
        if exists(annotations_file):
            ner_annotations, rel_annotations = self.read_annotations(annotations_file)
            document.sentences = self.generate_sentences(tokens_by_sentence, ner_annotations, rel_annotations)
        else:
            document.sentences = self.generate_test_sentences(tokens_by_sentence)
        return document

    def read_folder(self, folder_path: str) -> List[Document]:
        result = []
        files = listdir(folder_path)
        for file in tqdm(files):
            file_name, extension = splitext(file)
            if extension == '.txt':
                document = self.read_document(file_name, folder_path)
                result.append(document)
        return result


class BertReader(RuREBusReader):
    """
    Reader for BERT models using tokenization from `huggingface\tokenizers`
    """

    def __init__(self, tokenizer: BertWordPieceTokenizer, debug_file: str, max_sent_len=100):
        super().__init__(debug_file)
        self.tokenizer = tokenizer
        self.max_len = max_sent_len

    def split_by_str(self, sentences: List[str], pattern) -> List[str]:
        new_sentences = []
        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence)
            if len(tokens.ids) > self.max_len:
                splitted = sentence.split(pattern)
                splitted = [sent for sent in splitted if len(sent) > 0]
                new_sentences.extend(splitted)
            else:
                new_sentences.append(sentence)
        return new_sentences

    def split_by_maxtokens(self, sentences: List[str]) -> List[str]:
        new_sentences = []
        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence)
            if len(tokens.ids) > self.max_len:
                split_offsets = []
                for i, offset in enumerate(tokens.offsets):
                    if i % (self.max_len - 10) == self.max_len - 11:
                        split_offsets.append((offset[0] - 1, i))
                last_idx = 0
                for idx in split_offsets:
                    new_sentences.append(sentence[last_idx:idx[0]])
                    last_idx = idx[0]
                new_sentences.append(sentence[last_idx:])
            else:
                new_sentences.append(sentence)
        return new_sentences

    def check_sentences(self, sentences: List[str]):

        flag = True
        with open('log5.txt','a', encoding='utf-8') as f:
            for sentence in sentences:
                tokens = self.tokenizer.encode(sentence)
                if len(tokens.ids) > self.max_len:
                    flag = False
                    f.write("%d ## %s\n ------------------------------------------\n" % (len(tokens.ids), sentence, ))
        return flag

    def split_sentences(self, text):
        """
        A sequence of sentences splitting to push data into BERT
        """
        # first split by classic method
        sentences = sent_tokenize(text, language='russian')

        # then split by semicolon
        sentences = self.split_by_str(sentences, ';')

        # then by 2x new line
        sentences = self.split_by_str(sentences, '\n\n')

        # then by 1x line
        sentences = self.split_by_str(sentences, '\n')

        # finally split by max len
        sentences = self.split_by_maxtokens(sentences)

        # check tokens length
        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence)
            assert(len(tokens.ids) <= self.max_len)
        return sentences

    def read_tokens(self, text_file: str) -> List[BertSentence]:
        result = []
        with open(text_file, encoding='utf-8') as f:
            last_index = 0
            data = f.read()
            sentences = self.split_sentences(data)
            for sentence in sentences:
                i = data.find(sentence, last_index)
                tokens = self.tokenizer.encode(sentence)
                result.append(BertSentence(offset=i, tokens=tokens))
                last_index = i + len(sentence) - 1
        return result

    def fill_entities(self, bert_sentences: List[BertSentence], ner_annotations: List[NerAnnotation],
                      rel_annotations: List[RelAnnotation]):
        for sentence in bert_sentences:
            tokens = sentence.tokens
            current_entity = None
            words, offsets = tokens.tokens, tokens.offsets
            for i, (word, offset) in enumerate(zip(words, offsets)):
                if offset[0] == offset[1]:
                    # special token
                    continue
                ann = self.find_entity(offset[0] + sentence.offset, ner_annotations)
                if ann is not None:
                    if current_entity is None:
                        # found new entity
                        current_entity = Entity(
                            idx=ann.idx, label=ann.type, start=i, end=i
                        )
                    else:
                        if current_entity.idx == ann.idx:
                            # shift end of current entity
                            current_entity.end = i
                        else:
                            # consecutive entities
                            sentence.entities.append(current_entity)
                            current_entity = Entity(
                                idx=ann.idx, label=ann.type, start=i, end=i
                            )
                else:
                    if current_entity is not None:
                        sentence.entities.append(current_entity)
                        current_entity = None
            if current_entity is not None:
                sentence.entities.append(current_entity)
            sentence.relations = self.process_relations(sentence.entities, rel_annotations)

    def read_document(self, file_name: str, folder_path: str) -> Document:
        document = Document(file_name)
        self.current_document = document
        text_file = join(folder_path, file_name + '.txt')
        annotations_file = join(folder_path, file_name + '.ann')

        sentences = self.read_tokens(text_file)
        if exists(annotations_file):
            ner_annotations, rel_annotations = self.read_annotations(annotations_file)
            self.fill_entities(sentences, ner_annotations, rel_annotations)

        document.sentences = sentences
        return document

    def read_folder(self, folder_path: str) -> List[Document]:
        result = []
        files = listdir(folder_path)
        for file in tqdm(files):
            file_name, extension = splitext(file)
            if extension == '.txt':
                document = self.read_document(file_name, folder_path)
                result.append(document)
        return result
