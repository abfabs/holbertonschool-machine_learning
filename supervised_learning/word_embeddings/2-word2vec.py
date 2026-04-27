#!/usr/bin/env python3
"""Module to train a Word2Vec model using gensim"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model
    """
    # sg=0 for CBOW, sg=1 for Skip-gram
    sg = 0 if cbow else 1

    # Initialize model without sentences first to ensure seed is applied
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # Build vocabulary
    model.build_vocab(sentences)

    # Train model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
