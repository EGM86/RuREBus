from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple, Pattern
from .span_utils import tags_to_spans, NER_TAG_REGEX


def f1_score(y_pred: List[str], y_true: List[str]) ->Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    for label in y_pred:
        if label in y_true:
            true_positives[label] += 1
            y_true.remove(label)
        else:
            false_positives[label] += 1

    for label in y_true:
        false_negatives[label] += 1

    return true_positives, false_positives, false_negatives


def f1_span_score(y_pred: List[str], y_true: List[str],
                  entity_regex: Pattern = NER_TAG_REGEX) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Calculates FP, FN and TP from lists of entities tokens similary to Conll SRL metrics

    Parameters
    ----------
    y_pred : List[str], required.
        The predicted class labels for a sequence.

    y_true : List[str], required.
        The ethalon class labels for a sequence.

    entity_regex: Pattern, optional.
        Regex fot entity matching.
    Returns
    -------
    counters : Tuple[DefaultDict[int]]
        Tuple with TP, FP, FN counters dictionaries by entity type.

    """
    y_pred_spans = tags_to_spans(y_pred, entity_regex)
    y_true_spans = tags_to_spans(y_true, entity_regex)
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    for span in y_pred_spans:
        if span in y_true_spans:
            true_positives[span[0]] += 1
            y_true_spans.remove(span)
        else:
            false_positives[span[0]] += 1

    for span in y_true_spans:
        false_negatives[span[0]] += 1

    return true_positives, false_positives, false_negatives
