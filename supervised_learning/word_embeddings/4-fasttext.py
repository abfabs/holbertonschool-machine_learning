#!/usr/bin/env python3
"""
NLP - Word Embeddings - Task 4

Function to create, build, and train a Gensim FastText model.
"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim FastText model.

    Parameters
    ----------
    sentences : list of list of str
        Training corpus, where each sentence is a list of tokens.
    vector_size : int, optional
        Dimensionality of the embedding vectors.
    min_count : int, optional
        Minimum number of occurrences for a word to be included.
    negative : int, optional
        Number of negative samples used in negative sampling.
    window : int, optional
        Maximum distance between the current and predicted word.
    cbow : bool, optional
        If True, use CBOW training (sg=0); otherwise use Skip-gram (sg=1).
    epochs : int, optional
        Number of training iterations over the corpus.
    seed : int, optional
        Seed for the random number generator.
    workers : int, optional
        Number of worker threads to use during training.

    Returns
    -------
    gensim.models.FastText
        The trained FastText model.
    """
    # cbow=True → sg=0 (CBOW), cbow=False → sg=1 (Skip-gram)
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
        workers=workers,
    )

    return model