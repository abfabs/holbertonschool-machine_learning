#!/usr/bin/env python3
"""Module for TF-IDF embedding"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix
    """
    processed_sentences = []
    for sentence in sentences:
        # Standardize: lowercase and handle possessives/punctuation
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

    s = len(sentences)
    f = len(features)
    
    # Initialize matrices
    tf = np.zeros((s, f))
    df = np.zeros(f)
    
    feature_index = {word: i for i, word in enumerate(features)}

    # Calculate TF and DF
    for i, sentence_words in enumerate(processed_sentences):
        if not sentence_words:
            continue
        
        # Track words seen in this document for DF calculation
        words_in_doc = set()
        
        for word in sentence_words:
            if word in feature_index:
                idx = feature_index[word]
                tf[i, idx] += 1
                words_in_doc.add(idx)
        
        # Normalize TF by total words in the sentence
        # Note: In some variants, TF is just the raw count. 
        # But for the 1.0 values in your example, we use raw counts
        # or a specific normalization.
        
        for idx in words_in_doc:
            df[idx] += 1

    # Calculate IDF: ln(N / df)
    # Using the natural log as is standard in many ML frameworks
    idf = np.log(s / df)
    
    # Calculate TF-IDF
    # We must handle division by zero for words not in the sentences
    embeddings = tf * idf
    
    # L2 Normalization (Euclidean) per row
    # Based on the output 0.707 (which is 1/sqrt(2)), the result is L2 normalized
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    embeddings = np.divide(embeddings, norm, out=np.zeros_like(embeddings),
                           where=norm != 0)

    return embeddings, np.array(features)
