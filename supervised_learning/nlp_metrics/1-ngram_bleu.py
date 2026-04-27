#!/usr/bin/env python3
"""Calculates the n-gram BLEU score for a sentence."""
from collections import Counter
import math


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a candidate sentence.

    Parameters:
        references: list of reference translations, each a list of words.
        sentence: list containing the proposed sentence words.
        n: size of the n-gram to use for evaluation.

    Return:
        The n-gram BLEU score.
    """
    if len(sentence) < n:
        return 0

    # Extract n-grams from sentence
    sentence_ngrams = []
    for i in range(len(sentence) - n + 1):
        sentence_ngrams.append(tuple(sentence[i:i + n]))

    if len(sentence_ngrams) == 0:
        return 0

    # Count sentence n-grams
    sentence_counts = Counter(sentence_ngrams)

    # Extract n-grams from each reference and get max counts
    max_counts = Counter()
    for reference in references:
        ref_ngrams = []
        for i in range(len(reference) - n + 1):
            ref_ngrams.append(tuple(reference[i:i + n]))
        ref_counts = Counter(ref_ngrams)
        for ngram, count in ref_counts.items():
            max_counts[ngram] = max(max_counts[ngram], count)

    # Calculate clipped precision
    clipped = 0
    for ngram, count in sentence_counts.items():
        clipped += min(count, max_counts.get(ngram, 0))

    precision = clipped / len(sentence_ngrams)

    # Brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(ref_lengths,
                          key=lambda r: (abs(r - len(sentence)), r))

    if len(sentence) > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / len(sentence))

    return bp * precision
