from typing import List, Set, Tuple, Pattern
from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan
import re

NER_TAG_REGEX = re.compile(r'(?P<span_tag>[SBIE])-(?P<entity>\w+)')


def tags_to_spans(tag_sequence: List[str], tag_regex: Pattern) -> List[TypedStringSpan]:
    """
    Given a sequence of BIES tags with properties, extracts spans
    Tags template:
    OUT or {S,B,I,E}-Entity:Property
    S - one-token entity
    B - begin of entity
    I - inside entity
    E - end of entity

    Parameters
    ----------
    tag_sequence : List[str], required.
        The integer class labels for a sequence.

    tag_regex: Pattern, required.
        Regex for entities matching

    Returns
    -------
    spans : List[TypedStringSpan]
        The typed, extracted spans from the sequence, in the format (entity, (span_start, span_end)).
    """

    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # entity label should math {S,B,I,E}-Entity:Property template
        m = tag_regex.match(string_tag)
        if m is not None:
            span_tag = m.groupdict()['span_tag']
            entity = m.groupdict()['entity']
            if span_tag == 'B':
                # entering new span
                if active_tag is not None:
                    spans.add((active_tag, (span_start, span_end)))
                active_tag = entity
                span_start = index
                span_end = index
            elif span_tag == 'S':
                # entity with one token
                if active_tag is not None:
                    # add existing span
                    spans.add((active_tag, (span_start, span_end)))
                # also add current one-token entity
                active_tag, span_start, span_end = entity, index, index
                spans.add((active_tag, (span_start, span_end)))
                active_tag = None
            elif span_tag == 'E':
                # end of span
                if active_tag == entity:
                    # finish current span
                    span_end = index
                    spans.add((active_tag, (span_start, span_end)))
                else:
                    # unexpected: just make span with one token
                    if active_tag is not None:
                        # add existing span
                        spans.add((active_tag, (span_start, span_end)))
                    # also add current entity
                    active_tag, span_start, span_end = entity, index, index
                    spans.add((active_tag, (span_start, span_end)))
                active_tag = None
            elif span_tag == 'I':
                if active_tag == entity:
                    # inside span
                    span_end += 1
                else:
                    # unexpected: assume that this is begin of another span
                    if active_tag is not None:
                        spans.add((active_tag, (span_start, span_end)))
                    active_tag = entity
                    span_start = index
                    span_end = index
            else:
                assert False, "Unexpected case"
        else:
            # The span has ended
            if active_tag is not None:
                spans.add((active_tag, (span_end, span_end)))
            active_tag = None

    # Last token might have been a part of a valid span.
    if active_tag is not None:
        spans.add((active_tag, (span_start, span_end)))

    return list(spans)


