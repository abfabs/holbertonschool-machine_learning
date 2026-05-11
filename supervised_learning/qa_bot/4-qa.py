#!/usr/bin/env python3
"""Multi-reference Question Answering bot."""

qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search

EXIT_WORDS = {"exit", "quit", "goodbye", "bye"}


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts.

    corpus_path: path to the corpus of reference documents
    """
    while True:
        question = input("Q: ")

        if question.strip().lower() in EXIT_WORDS:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        answer = qa(question, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))