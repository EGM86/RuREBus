from random import random, choice, shuffle
from typing import List

from .fields import Sentence, RelationSample, Document


class RelationsSampler:
    """
    Base class for creating samples from sentences
    """
    def sample_sentence(self, sentence: Sentence) -> List[RelationSample]:
        raise NotImplemented()

    def sample_document(self, document: Document) -> List[RelationSample]:
        samples = []
        for sentence in document.sentences:
            samples.extend(self.sample_sentence(sentence))
        return samples

    def sample_data(self, documents: List[Document]) -> List[RelationSample]:
        samples = []
        for document in documents:
            samples.extend(self.sample_document(document))
        return samples


class WholeSampler(RelationsSampler):
    """
    Samples all possible pairs of entities in a sentence
    """
    def sample_sentence(self, sentence: Sentence) -> List[RelationSample]:
        samples = []  # type: List[RelationSample]
        # entities should be already sorted
        entities = sentence.entities

        # generate all possible pairs
        for i in range(len(entities) - 1):
            for j in range(i + 1, len(entities)):
                e1 = entities[i]
                e2 = entities[j]
                rel_type = 'OUT'
                for rel in sentence.relations:
                    if (rel.arg1 == i and rel.arg2 == j) or \
                            (rel.arg2 == i and rel.arg1 == j):
                        rel_type = rel.type
                        break
                samples.append(RelationSample(tokens=sentence.tokens, e1=e1, e2=e2, relation=rel_type))
        return samples


class PartialSampler(RelationsSampler):
    """
    Creates negative samples by heuristic that numbers of positive and negative samples might be same
    """
    def sample_sentence(self, sentence: Sentence) -> List[RelationSample]:
        positive_samples = []
        if len(sentence.entities) == 0:
            return []
        for rel in sentence.relations:
            e1 = sentence.entities[rel.arg1]
            e2 = sentence.entities[rel.arg2]
            positive_samples.append(
                RelationSample(tokens=sentence.tokens, e1=e1, e2=e2, relation=rel.type)
            )

        negative_samples = []
        max_tries = max(5, len(positive_samples)*2)
        tries = 0
        while tries < max_tries:
            tries += 1
            e1 = choice(sentence.entities)
            e2 = choice(sentence.entities)

            if e1.idx == e2.idx:
                continue

            for rel in positive_samples:
                if (e1 == rel.e1 and e2 == rel.e2) or (e1 == rel.e2 and e2 == rel.e1):
                    continue

            negative_samples.append(
                RelationSample(tokens=sentence.tokens, e1=e1, e2=e2, relation='OUT')
            )

        result = positive_samples + negative_samples
        if len(result) == 0:
            return []

        shuffle(result)
        return result
