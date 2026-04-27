#!/usr/bin/env python3
"""Calculates the cumulative n-gram BLEU score for a sentence."""
from collections import Counter
import math


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a candidate sentence.

    Parameters:
        references: list of reference translations, each a list of words.
        sentence: list containing the proposed sentence words.
        n: size of the largest n-gram to use for evaluation.

    Return:
        The cumulative n-gram BLEU score.
    """
    if len(sentence) == 0:
        return 0

    # Calculate individual n-gram BLEU scores for n=1 to n=N
    bleu_scores = []
    for k in range(1, n + 1):
        # Extract k-grams from sentence
        sentence_kgrams = []
        for i in range(len(sentence) - k + 1):
            sentence_kgrams.append(tuple(sentence[i:i + k]))

        if len(sentence_kgrams) == 0:
            bleu_scores.append(0)
            continue

        # Count sentence k-grams
        sentence_counts = Counter(sentence_kgrams)

        # Extract k-grams from each reference and get max counts
        max_counts = Counter()
        for reference in references:
            ref_kgrams = []
            for i in range(len(reference) - k + 1):
                ref_kgrams.append(tuple(reference[i:i + k]))
            ref_counts = Counter(ref_kgrams)
            for kgram, count in ref_counts.items():
                max_counts[kgram] = max(max_counts[kgram], count)

        # Calculate clipped precision for k-gram
        clipped = 0
        for kgram, count in sentence_counts.items():
            clipped += min(count, max_counts.get(kgram, 0))

        precision_k = clipped / len(sentence_kgrams)

        # Brevity penalty (same for all n-grams)
        ref_lengths = [len(ref) for ref in references]
        closest_ref_len = min(ref_lengths,
                              key=lambda r: (abs(r - len(sentence)), r))

        if len(sentence) > closest_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - closest_ref_len / len(sentence))

        bleu_scores.append(bp * precision_k)

    # Geometric mean: all n-gram scores weighted evenly
    if len(bleu_scores) == 0:
        return 0

    # Cumulative BLEU = exp(average(ln(p_k))) for k=1 to n
    log_sum = sum(math.log(p) for p in bleu_scores if p > 0)
    avg_log = log_sum / len(bleu_scores)
    return math.exp(avg_log)
