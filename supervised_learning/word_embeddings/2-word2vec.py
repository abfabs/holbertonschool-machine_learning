#!/usr/bin/env python3
"""
Function to create, build, and train a Word2Vec model using Gensim
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    """
    # Training algorithm
    sg = 0 if cbow else 1

    # Create the Word2Vec model
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

    # Model vocabulary
    model.build_vocab(sentences)

    # Training
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
