#!/usr/bin/env python3
"""Function to create, build, and train a FastText model using Gensim"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim FastText model.

    sentences: list of sentences for training
    vector_size: dimensionality of the embedding vectors
    min_count: minimum word occurrences for training
    window: maximum distance between current and predicted word
    negative: size of negative sampling
    cbow: True for CBOW, False for Skip-gram
    epochs: number of training iterations
    seed: random number generator seed
    workers: number of worker threads

    Returns: trained model
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
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
