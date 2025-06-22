# Goal: Using FAISS and our precomputed index, find related topics to a given topic.

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_related_index(index_path: str, titles_path: str):
    """
    Loads a FAISS index and its corresponding list of Wikipedia article titles.
    """
    index = faiss.read_index(index_path)
    with open(titles_path, "rb") as f:
        titles = pickle.load(f)
    return index, titles


def find_related_topics(query: str, model: SentenceTransformer, index, titles, top_k=10):
    """
    Embeds the query and retrieves the top-k most similar Wikipedia topics from the index.
    """
    query_embedding = model.encode(query).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    related = []
    for i in indices[0]:
        if 0 <= i < len(titles):
            related.append(titles[i])
    return related