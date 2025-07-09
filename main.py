# Wrapper that takes input (prompt) and returns output (answer)
import sys
import cohere
import wikipedia
from sentence_transformers import SentenceTransformer, util
from src.wikipedia_search import fetch_article, chunk_text
from src.embedder import embed_chunks, build_index
from src.retriever import retrieve
from src.answer import generate_answer
from src.related import load_related_index, find_related_topics

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

from dotenv import load_dotenv
import os
import torch

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize Cohere client and SentenceTransformer model
co = cohere.Client(API_KEY)

# Maps english sentences to n-dimensional vectors
ASK_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu') # n = 384, decent quality but quicker
FIND_MODEL = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu') # n = 768, best quality but slower

# Load FAISS related-topic index and titles
RELATED_INDEX_PATH = "binaries/wiki_best_summary_index.faiss"
RELATED_TITLES_PATH = "binaries/wiki_best_summary_titles.pkl"

def ask_question():
    """
    Ask a question to Wikipedia and return an answer."""
    question = input("Ask wikipedia anything: ")
    search_results = wikipedia.search(question, results = 10, suggestion = False)
    if not search_results:
        print("No results found.")
        exit()

    # embed articles and user query
    title_embeddings = ASK_MODEL.encode(search_results, convert_to_tensor=True)
    query_embedding = ASK_MODEL.encode(question, convert_to_tensor=True)

    # rank titles and obtain the top 4
    cosine_scores = util.pytorch_cos_sim(query_embedding, title_embeddings)[0]
    k = min(4, len(search_results))
    scores = cosine_scores.topk(k)
    top_scores = scores.values
    top_indices = scores.indices

    top_titles= [search_results[top_indices[i]] for i in range(len(top_indices)) if (top_scores[i] > 0.4)]

    # Only consider real wikipedia articles
    all_chunks = []
    valid_articles = []
    for title in top_titles:
        try: 
            article = fetch_article(title)
            chunks = chunk_text(article)
            all_chunks.extend(chunks)
            valid_articles.append(title)
        except Exception as e:
            continue

    # If no valid articles were found, exit
    if not all_chunks:
        print("No valid articles found.")
        exit()

    # Build FAISS index and retrieve top chunks
    index = build_index(embed_chunks(all_chunks, ASK_MODEL))
    top_chunks = retrieve(question, all_chunks, index, ASK_MODEL)
    response = generate_answer(question, top_chunks, co)

    print("Answer:", response)
    print("Wikipedia Sources:", valid_articles)

def find_related(num_children, num_grandchildren):
    """
    Find related topics to a given topic.
    """
    related_index, related_titles = load_related_index(RELATED_INDEX_PATH, RELATED_TITLES_PATH)

    topic = input("Enter a topic or phrase to find related Wikipedia concepts: ")
    results = find_related_topics(topic, FIND_MODEL, related_index, related_titles, top_k=num_children)

    related_topics_dict = {topic: results}
    for r in results:
        related_topics_dict[r] = find_related_topics(topic + " and " + r, FIND_MODEL, related_index, related_titles, top_k=num_grandchildren+1)[1:]

    def print_topic_tree(root, related_topics_dict, indent="", last=True):
        """    
        Print the topic tree in a clean and structured format.
        """
        print(root)
        children = related_topics_dict.get(root, [])
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            prefix = "└── " if is_last_child else "├── "
            print(prefix + child)

            grandchildren = related_topics_dict.get(child, [])
            for j, grandchild in enumerate(grandchildren):
                is_last_grandchild = (j == len(grandchildren) - 1)
                sub_prefix = "    " if is_last_child else "│   "
                connector = "└── " if is_last_grandchild else "├── "
                print(sub_prefix + connector + grandchild)

    print("\nRelated Topics:")
    print_topic_tree(topic, related_topics_dict)

if __name__ == "__main__":

    #if len(sys.argv) != 2, "enter \"ask\" to ask wikipedia a question or \"find\" to find related topics"

    choice = sys.argv[1] if len(sys.argv) == 2 else None

    if choice == "ask":
        ask_question()
    elif choice == "find":
        find_related(4, 3)
    else:
        print("in the command line")
        print(" enter \"ask\" to ask wikipedia a question or;") 
        print(" enter \"find\" to find related topics")
