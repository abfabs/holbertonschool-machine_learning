#!/usr/bin/env python3
"""Function to create, build, and train a Word2Vec model using Gensim"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    sentences:   list of sentences (list of word lists) for training
    vector_size: dimensionality of the embedding vectors
    min_count:   minimum word frequency threshold
    window:      max distance between current and predicted word
    negative:    number of negative samples
    cbow:        True -> CBOW (sg=0), False -> Skip-gram (sg=1)
    epochs:      number of training iterations
    seed:        random number generator seed
    workers:     number of worker threads

    Returns: trained Word2Vec model
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        iter=epochs,
        seed=seed,
        workers=workers
    )

    return model
