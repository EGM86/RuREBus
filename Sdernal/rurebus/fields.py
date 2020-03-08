from typing import NamedTuple, List, Union

from tokenizers import Encoding


class Token(NamedTuple):
    """
    A token representation with features
    Contains:
    - offset in characters from the begging of the document
    - a word string
    """
    offset: int = None
    word: str = None
    pos_tag: str = None

    def __str__(self):
        return self.word

    def __repr__(self):
        return str(self)


class Entity:
    """
    An entity representation with it's span
    Contains:
    - start and end of the span
    - index of entity in data
    - entity label
    """
    def __init__(self, start: int, end: int, label: str, idx: int):
        self.start = start
        self.end = end
        self.label = label
        self.idx = idx

    def __str__(self):
        return 'T%d\t%s %d\t%d' % (self.idx, self.label, self.start, self.end)

    def __repr__(self):
        return str(self)


class Relation(NamedTuple):
    """
    A relation representation
    Contains:
    - index of relation
    - type of relation
    - indexes of related entities
    """
    idx: int = None
    type: str = None
    arg1: int = None  # index in entities list
    arg2: int = None  # index in entities list

    def __str__(self):
        return 'R%d\t%s %d %d' % (self.idx, self.type, self.arg1, self.arg2)

    def __repr__(self):
        return str(self)


class Sentence:
    """
    A simple container of tokens, entities and relations
    Contains:
     - list of tokens
     - list of entities
     - list of relations
    """
    def __init__(self):
        self.tokens = []   # type: List[Token]
        self.entities = []  # type: List[Entity]
        self.relations = []  # type: List[Relation]


class RelationSample(NamedTuple):
    """
    A Sample for relation extraction
    Contains list of tokens(or Bert encoding), pair of entities and type of relation
    """
    tokens: Union[List[Token], Encoding] = None
    e1: Entity = None
    e2: Entity = None
    relation: str = None


class BertSentence:
    """
    Container for sentence like `Sentence` but for BERT tokens
    """
    def __init__(self, offset: int, tokens: Encoding):
        self.offset = offset
        self.tokens = tokens  # type: Encoding
        self.entities = []  # type: List[Entity]
        self.relations = []  # type: List[Relation]


class Document:
    """
    A list of sentences and name of file
    """
    def __init__(self, file_name):
        self.sentences = []  # type: Union[List[Sentence], List[BertSentence]]
        self.file_name = file_name


class NerAnnotation(NamedTuple):
    idx: int = None
    type: str = None
    start: int = None
    end: int = None
    value: str = None

    def __str__(self):
        return 'T%d\t%s %d %d\t%s' % (self.idx, self.type, self.start, self.end, self.value)

    def __repr__(self):
        return str(self)


class RelAnnotation(NamedTuple):
    idx: int = None
    type: str = None
    arg1: int = None
    arg2: int = None

    def __str__(self):
        return 'R%d\t%s %d %d' % (self.idx, self.type, self.arg1, self.arg2)

    def __repr__(self):
        return str(self)


