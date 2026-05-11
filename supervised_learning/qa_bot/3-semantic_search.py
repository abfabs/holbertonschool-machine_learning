#!/usr/bin/env python3
"""Semantic search on a corpus of documents using Universal Sentence Encoder."""

import os
import numpy as np
import tensorflow_hub as hub


model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    corpus_path: path to the corpus of reference documents
    sentence: sentence from which to perform semantic search
    Returns: reference text of the most similar document
    """
    documents = []
    doc_texts = []

    for filename in sorted(os.listdir(corpus_path)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(corpus_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        doc_texts.append(text)
        documents.append(text)

    if not documents:
        return None

    all_texts = [sentence] + documents
    embeddings = model(all_texts).numpy()

    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(
        doc_embeddings, axis=1, keepdims=True
    )

    similarities = np.dot(doc_norms, query_norm)
    best_idx = np.argmax(similarities)

    return documents[best_idx]