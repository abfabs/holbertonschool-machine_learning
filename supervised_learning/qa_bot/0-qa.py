#!/usr/bin/env python3
"""Question answering with BERT from TF Hub."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer a question."""
    if not question or not reference:
        return None

    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf",
        truncation=True,
        max_length=512
    )

    input_word_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]
    input_type_ids = inputs["token_type_ids"]

    outputs = model([
        input_word_ids,
        input_mask,
        input_type_ids
    ])

    start_logits, end_logits = outputs[0], outputs[1]
    start = tf.argmax(start_logits[0]).numpy()
    end = tf.argmax(end_logits[0]).numpy()

    if start >= end:
        return None

    tokens = input_word_ids[0][start:end + 1]
    answer = tokenizer.decode(tokens)

    if answer in ("[CLS]", "[SEP]"):
        return None

    return answer