#!/usr/bin/env python3

import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    
    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis
               If None, all words within sentences should be used
               
    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    # 1. Preprocess sentences: lowercase and filter words
    processed_sentences = []
    all_words = []
    
    for sentence in sentences:
        # Lowercase and replace punctuation with space or remove it
        # This regex handles possessives (children's -> children) and punctuation
        words = re.findall(r'\b\w+\b', sentence.lower())
        processed_sentences.append(words)
        all_words.extend(words)

    # 2. Define vocabulary (features)
    if vocab is None:
        # Get unique words and sort alphabetically
        features = sorted(list(set(all_words)))
    else:
        features = vocab

    # Map words to indices for faster lookup
    vocab_dict = {word: i for i, word in enumerate(features)}
    
    # 3. Create embedding matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(processed_sentences):
        for word in words:
            if word in vocab_dict:
                embeddings[i, vocab_dict[word]] += 1

    return embeddings, features
