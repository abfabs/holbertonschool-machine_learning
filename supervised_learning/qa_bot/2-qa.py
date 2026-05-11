#!/usr/bin/env python3
"""QA bot loop."""

exit_words = {"exit", "quit", "goodbye", "bye"}

while True:
    question = input("Q: ")
    if question.strip().lower() in exit_words:
        print("A: Goodbye")
        break
    print("A: ")
