# Wrapper file that takes input (prompt) and returns output (answer)
import cohere
import wikipedia
from sentence_transformers import SentenceTransformer, util
from src.wikipedia_search import fetch_article, chunk_text
from src.embedder import embed_chunks, build_index
from src.retriever import retrieve
from src.answer import generate_answer
from src.related import load_related_index, find_related_topics


from dotenv import load_dotenv
import os
import torch

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize Cohere client and SentenceTransformer model
co = cohere.Client(API_KEY)
# Maps english sentences to 384-dimensional vectors
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

# Load FAISS related-topic index and titles
RELATED_INDEX_PATH = "binaries/wiki_summary_index.faiss"
RELATED_TITLES_PATH = "binaries/wiki_summary_titles.pkl"
#RELATED_INDEX_PATH = "binaries/wiki_longer_summary_index.faiss"
#RELATED_TITLES_PATH = "binarieswiki_longer_summary_titles.pkl"

related_index, related_titles = load_related_index(RELATED_INDEX_PATH, RELATED_TITLES_PATH)

def ask_question():
    """
    Ask a question to Wikipedia and return an answer."""
    question = input("Ask wikipedia anything: ")
    search_results = wikipedia.search(question, results = 10)
    if not search_results:
        print("No results found.")
        exit()

    # embed articles and user query
    title_embeddings = MODEL.encode(search_results, convert_to_tensor=True)
    query_embedding = MODEL.encode(question, convert_to_tensor=True)

    # rank titles and obtain the top 4
    cosine_scores = util.pytorch_cos_sim(query_embedding, title_embeddings)[0]
    k = min(4, len(search_results))
    top_indices = cosine_scores.topk(k).indices
    top_titles= [search_results[i] for i in top_indices]


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
            print(f"Skipping {title}: {e}")

    # If no valid articles were found, exit
    if not all_chunks:
        print("No valid articles found.")
        exit()

    # Build FAISS index and retrieve top chunks
    index = build_index(embed_chunks(all_chunks, MODEL))
    top_chunks = retrieve(question, all_chunks, index, MODEL)
    response = generate_answer(question, top_chunks, co)

    print("Answer:", response)
    print("Wikipedia Sources:", valid_articles)

def find_related():
    """
    Find related topics to a given topic.
    """
    topic = input("Enter a topic or phrase to find related Wikipedia concepts: ")
    results = find_related_topics(topic, MODEL, related_index, related_titles, top_k=10)

    print("\n Related Topics:")
    for r in results:
        print("-", r)


if __name__ == "__main__":

    print("Choose an option:")
    print("1. Ask a question")
    print("2. Find related topics")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        ask_question()
    elif choice == "2":
        find_related()
    else:
        print("Invalid option.")
