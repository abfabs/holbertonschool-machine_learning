#!/usr/bin/env python3
"""Module for Bag of Words embedding"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    """
    processed_sentences = []
    for sentence in sentences:
        # Lowercase and handle possessives (children's -> children)
        line = sentence.lower().replace("'s", "")
        # Replace non-alphabetic chars with spaces
        clean_line = "".join([c if c.isalpha() else " " for c in line])
        processed_sentences.append(clean_line.split())

    if vocab is None:
        all_words = []
        for s in processed_sentences:
            all_words.extend(s)
        # Use set for unique words, then sort alphabetically
        features = sorted(list(set(all_words)))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)
    feature_index = {word: i for i, word in enumerate(features)}

    for i, sentence_words in enumerate(processed_sentences):
        for word in sentence_words:
            if word in feature_index:
                embeddings[i, feature_index[word]] += 1

    return embeddings, np.array(features)
