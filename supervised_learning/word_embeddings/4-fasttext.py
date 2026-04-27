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
    """
    # Set the training algorithm
    sg = 0 if cbow else 1

    # Create the FastText model
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

    # prepare vocabulary
    model.build_vocab(sentences)

    # Train
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
