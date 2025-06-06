#Goal: Embed text chunks into vector representations and build a FAISS index 
#      for efficient similarity search.
import faiss
from sentence_transformers import SentenceTransformer 


def embed_chunks(chunks, MODEL):
    """
    Turns text chunks into embeddings using a local transformer model.
    Returns a matrix where each row represents a vector embedding of a chunk.
    """
    embeddings = MODEL.encode(chunks, convert_to_numpy=True)
    return embeddings

def build_index(embeddings):
    """
    Creates a FAISS L2 (Euclidean norm) index from the embeddings.
    This index can be used to efficiently search for nearest neighbors.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index