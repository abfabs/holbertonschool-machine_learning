#!/usr/bin/env python3
"""Module to train a Word2Vec model using gensim"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model
    """
    # Determine training algorithm: sg=0 for CBOW, sg=1 for Skip-gram
    if cbow is True:
        sg = 0
    else:
        sg = 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
