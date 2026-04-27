#!/usr/bin/env python3
"""
NLP - Word Embeddings - Task 3

Utility to convert a trained Gensim Word2Vec model into an Embedding layer.
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a Gensim Word2Vec model into a Keras Embedding layer.

    The returned layer uses the learned word vectors from `model.wv` as
    its weight matrix, so that token indices produced with the same
    vocabulary ordering can be fed directly into this embedding layer.

    Parameters
    ----------
    model : gensim.models.Word2Vec
        A trained Gensim Word2Vec model whose embeddings will initialize
        the Keras Embedding layer.

    Returns
    -------
    tf.keras.layers.Embedding
        A Keras Embedding layer initialized with the Word2Vec weights.
        The layer is trainable, so the embeddings can be fine-tuned
        during downstream model training.
    """
    keys = model.wv
    weights = keys.vectors

    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )
