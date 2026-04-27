#!/usr/bin/env python3
"""
Creates, builds, and trains a Gensim FastText model.
"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Trains a FastText model on a list of tokenized sentences.

    Parameters:
        sentences: list of sentences to be trained on.
        vector_size: dimensionality of the embedding layer.
        min_count: minimum number of occurrences of a word for training.
        window: maximum distance between the current and predicted word.
        negative: size of negative sampling.
        cbow: True for CBOW, False for Skip-gram.
        epochs: number of iterations to train over.
        seed: seed for the random number generator.
        workers: number of worker threads to train the model.

    Return:
        The trained FastText model.
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
