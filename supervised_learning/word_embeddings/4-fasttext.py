#!/usr/bin/env python3
"""
Utility to create, build, and train a Gensim FastText model.
"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
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
    window : int, optional
        Maximum distance between the current and predicted word.
    negative : int, optional
        Number of negative samples used in training.
    cbow : bool, optional
        If True, use CBOW training; otherwise use Skip-gram.
    epochs : int, optional
        Number of training iterations.
    seed : int, optional
        Seed for the random number generator.
    workers : int, optional
        Number of worker threads to use during training.

    Returns
    -------
    gensim.models.FastText
        The trained FastText model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=epochs
    )

    return model
