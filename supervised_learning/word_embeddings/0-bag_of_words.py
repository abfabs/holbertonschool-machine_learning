#!/usr/bin/env python3
"""Module for Bag of Words embedding"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    """
    processed_sentences = []
    
    for sentence in sentences:
        # 1. Lowercase
        # 2. Replace characters like '!' or '.' with space
        # 3. Specifically handle "children's" by replacing "'s" with space
        #    before splitting, or simply filter out single 's'
        line = sentence.lower().replace("'s", "")
        
        # Clean non-alphabetic characters (except spaces)
        words = ""
        for char in line:
            if char.isalpha() or char.isspace():
                words += char
            else:
                words += " "
        
        processed_sentences.append(words.split())

    if vocab is None:
        # Build vocab from all words found
        all_words = []
        for s in processed_sentences:
            all_words.extend(s)
        features = sorted(list(set(all_words)))
    else:
        features = vocab

    # Create the matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)
    
    feature_index = {word: i for i, word in enumerate(features)}

    for i, sentence_words in enumerate(processed_sentences):
        for word in sentence_words:
            if word in feature_index:
                embeddings[i, feature_index[word]] += 1

    return embeddings, np.array(features)
