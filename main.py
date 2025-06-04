# CLI - Command Line Interface
# Is a wrapper that takes input (prompt) and returns output (answer)

from src.pipeline import answer_question

if __name__ == "main":

    query = input("Ask Anything: ")
    answer = answer_question(query)

    print(answer)


