# Goal: Retrieve relevant chunks from a FAISS index based on a query.

def embed_query(query, MODEL):
    """
    Converts a query string into a vector embedding using the provided model.
    """
    return MODEL.encode([query], convert_to_numpy=True)[0]

def retrieve(query, chunks, index, MODEL, top_k=3):
    """
    Retrieves the top_k most relevant chunks from the FAISS index based on the query.
    
    Parameters:
    - query: The input question or query string.
    - model: The model used to embed the query.
    - chunks: The list of text chunks to search through.
    - index: The FAISS index containing embeddings of the chunks.
    - top_k: The number of top relevant chunks to return.
    
    Returns:
    - A list of tuples containing the indices and distances of the top_k relevant chunks.
    """
    # reshape so that we have one row (effectively a 2D array with one sample)
    query_vec = embed_query(query, MODEL).reshape(1, -1)
    indices = index.search(query_vec, top_k)[1]

    return [chunks[i] for i in indices[0]]
    