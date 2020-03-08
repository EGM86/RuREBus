import string
from typing import List

from tqdm import tqdm

from .vocab import Vocab
from .fields import Document, Sentence, BertSentence, RelationSample
from pymorphy2.tagset import OpencorporaTag


class BertIndexer:
    """
    fills Ner and Rel labels vocab snd generate labels of sentence for NER and RE
    """
    def __init__(self):
        self.ner_vocab = Vocab()
        self.ner_vocab.add('OUT')
        self.rel_vocab = Vocab(paddings=False)
        self.rel_vocab.add('OUT')

    def index_sentence(self, sentence: BertSentence):
        """
        Fills vocabs
        """
        for entity in sentence.entities:
            self.ner_vocab.fill(['S-' + entity.label,
                                 'B-' + entity.label,
                                 'I-' + entity.label,
                                 'E-' + entity.label])

        for relation in sentence.relations:
            self.rel_vocab.add(relation.type)

    def index_documents(self, documents: List[Document]):
        for documents in tqdm(documents):
            for sentence in documents.sentences:
                self.index_sentence(sentence)

    def get_ner_labels(self, sentence: BertSentence):
        """
        Returns indexes of ner labels for sentence tokens
        """
        sentence_len = len(sentence.tokens.ids)

        entity_indices = [self.ner_vocab['OUT']] * sentence_len

        for entity in sentence.entities:
            start, end = entity.start, entity.end
            if start == end:
                entity_indices[start] = self.ner_vocab['S-' + entity.label]
            else:
                entity_indices[start] = self.ner_vocab['B-' + entity.label]
                entity_indices[end] = self.ner_vocab['E-' + entity.label]
                for i in range(start + 1, end):
                    entity_indices[i] = self.ner_vocab['I-' + entity.label]

        return entity_indices

    def get_rel_labels(self, sample: RelationSample):
        """
        like  `get_ner_labels` but also returns label of relation in sample
        """
        sentence_len = len(sample.tokens.ids)
        entity_indices = [self.ner_vocab['OUT']] * sentence_len

        for entity in [sample.e1, sample.e2]:
            start, end = entity.start, entity.end
            if start == end:
                entity_indices[start] = self.ner_vocab['S-' + entity.label]
            else:
                entity_indices[start] = self.ner_vocab['B-' + entity.label]
                entity_indices[end] = self.ner_vocab['E-' + entity.label]
                for i in range(start + 1, end):
                    entity_indices[i] = self.ner_vocab['I-' + entity.label]

        return entity_indices, self.rel_vocab[sample.relation]


class DocumentIndexer:
    """
    Indexer for non-BERT models
    """
    def __init__(self):
        self.word_vocab = Vocab(lowercase=True)
        self.relation_type_vocab = Vocab(paddings=False)
        self.ner_vocab = Vocab()
        self.ner_vocab.add('OUT')
        self.char_vocab = Vocab()
        self.case_vocab = Vocab()
        self.pos_vocab = Vocab()
        self.pos_vocab.fill(OpencorporaTag.PARTS_OF_SPEECH)
        self.pos_vocab.add('PUNCT')
        self.relation_type_vocab.add('OUT')

    def index_documents(self, documents: List[Document]):
        for document in tqdm(documents):
            for sentence in document.sentences:
                self.index_sentence(sentence)

    def index_sentence(self, sentence: Sentence):
        for token in sentence.tokens:
            self.word_vocab.add(token.word)
            capitalization = get_capitalization_template(token.word)
            self.case_vocab.add(capitalization)
            for char in token.word:
                self.char_vocab.add(char)

        for entity in sentence.entities:
            self.ner_vocab.fill(['S-' + entity.label,
                                 'B-' + entity.label,
                                 'I-' + entity.label,
                                 'E-' + entity.label])

        for rel in sentence.relations:
            self.relation_type_vocab.add(rel.type)

    def sentence_to_indexes(self, sentence: Sentence):
        words = [token.word for token in sentence.tokens]
        word_indices = [self.word_vocab[word] for word in words]
        case_indices = [self.case_vocab[get_capitalization_template(word)] for word in words]
        entity_indices = [self.ner_vocab['OUT']] * len(sentence.tokens)
        pos_indices = [self.pos_vocab[token.pos_tag] for token in sentence.tokens]
        char_indices = []
        for word in words:
            word_chars = [self.char_vocab[char] for char in word]
            char_indices.append(word_chars)

        for entity in sentence.entities:
            start, end = entity.start, entity.end
            if start == end:
                entity_indices[start] = self.ner_vocab['S-' + entity.label]
            else:
                entity_indices[start] = self.ner_vocab['B-' + entity.label]
                entity_indices[end] = self.ner_vocab['E-' + entity.label]
                for i in range(start+1, end):
                    entity_indices[i] = self.ner_vocab['I-' + entity.label]

        return {'words': word_indices, 'chars': char_indices, 'cases': case_indices, 'pos': pos_indices, 'ner': entity_indices}

    def sample_to_indexes(self, sample: RelationSample):
        words = [token.word for token in sample.tokens]
        word_indices = [self.word_vocab[word] for word in words]
        case_indices = [self.case_vocab[get_capitalization_template(word)] for word in words]
        entity_indices = [self.ner_vocab['OUT']] * len(sample.tokens)
        pos_indices = [self.pos_vocab[token.pos_tag] for token in sample.tokens]
        char_indices = []
        for word in words:
            word_chars = [self.char_vocab[char] for char in word]
            char_indices.append(word_chars)
        relation_index = self.relation_type_vocab[sample.relation]

        # fill entities features
        e1 = sample.e1
        e2 = sample.e2

        for entity in [e1, e2]:
            start, end = entity.start, entity.end
            if start == end:
                entity_indices[start] = self.ner_vocab['S-' + entity.label]
            else:
                entity_indices[start] = self.ner_vocab['B-' + entity.label]
                entity_indices[end] = self.ner_vocab['E-' + entity.label]
                for i in range(start + 1, end):
                    entity_indices[i] = self.ner_vocab['I-' + entity.label]

        return {'words': word_indices, 'chars': char_indices, 'cases': case_indices, 'pos': pos_indices,
                'ner': entity_indices, 'relation': relation_index}


def get_capitalization_template(word: str) -> str:
    """ Capitalization template """
    template = ''
    for c in word:
        if c.isdigit():
            if template[-2:] != '11':
                template += '1'
        elif c.isspace():
            if template[-2:] != '__':
                template += '_'
        elif c.isupper() and c in string.ascii_letters:
            if template[-2:] != 'ZZ':
                template += 'Z'
        elif c.islower() and c in string.ascii_letters:
            if template[-2:] != 'zz':
                template += 'z'
        elif c.isupper() and c not in string.ascii_letters:
            if template[-2:] != 'ЯЯ':
                template += 'Я'
        elif c.islower() and c not in string.ascii_letters:
            if template[-2:] != 'яя':
                template += 'я'
        elif template[-2:] != '**':
            template += '*'
    return template