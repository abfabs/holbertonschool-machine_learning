#!/usr/bin/env python3
"""Calculates the cumulative n-gram BLEU score for a sentence."""
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
        ref_ngrams
