#!/usr/bin/env python3
"""Calculates the unigram BLEU score for a sentence."""
import math


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a candidate sentence.

    Parameters:
        references: list of reference translations, each a list of words.
        sentence: list containing the proposed sentence words.

    Return:
        The unigram BLEU score.
    """
    if len(sentence) == 0:
        return 0

    counts = {}
    for word in sentence:
        counts[word] = counts.get(word, 0) + 1

    max_counts = {}
    for reference in references:
        ref_counts = {}
        for word in reference:
            ref_counts[word] = ref_counts.get(word, 0) + 1
        for word, count in ref_counts.items():
            if word not in max_counts or count > max_counts[word]:
                max_counts[word] = count

    clipped = 0
    for word, count in counts.items():
        clipped += min(count, max_counts.get(word, 0))

    precision = clipped / len(sentence)

    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(ref_lengths,
                          key=lambda r: (abs(r - len(sentence)), r))

    if len(sentence) > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / len(sentence))

    return bp * precision
