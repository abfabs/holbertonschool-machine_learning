#!/usr/bin/env python3
"""Module for TF-IDF embedding"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix matching standard smoothed output
    """
    processed_sentences = []
    for sentence in sentences:
        # Handle possessives and clean punctuation
        line = sentence.lower().replace("'s", "")
        clean_line = "".join([c if c.isalpha() else " " for c in line])
        processed_sentences.append(clean_line.split())

    if vocab is None:
        all_words = []
        for s in processed_sentences:
            all_words.extend(s)
        features = sorted(list(set(all_words)))
    else:
        features = vocab

    n = len(sentences)
    f = len(features)
    feature_index = {word: i for i, word in enumerate(features)}

    # Initialize Term Frequency and Document Frequency
    tf = np.zeros((n, f))
    df = np.zeros(f)

    for i, words in enumerate(processed_sentences):
        # Calculate TF (counts)
        for word in words:
            if word in feature_index:
                tf[i, feature_index[word]] += 1
        # Calculate DF (binary presence)
        words_set = set(words)
        for j, word in enumerate(features):
            if word in words_set:
                df[j] += 1

    # Smoothed IDF: ln((1 + n) / (1 + df)) + 1
    idf = np.log((1 + n) / (1 + df)) + 1

    # Calculate TF-IDF
    embeddings = tf * idf

    # L2 Normalization: divide each row by its Euclidean norm
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Avoid division by zero for empty sentences
    embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings),
                           where=norms != 0)

    return embeddings, np.array(features)
