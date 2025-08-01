# Goal: Using the retrieved chunks, answer the query.

def generate_answer(query, chunks, cohere_client):
    """
    Generates an answer to the query using the retrieved chunks and a Cohere model.
    """
    prompt = f"""

    Using ONLY the following information, answer the question:
    Question: {query}
    Context: {chunks}
    Answer:"""

    response = cohere_client.chat(
        message = prompt,
        model="command-r",  # or whatever you have access to
        temperature=0.5)
    
    return response.text.strip()