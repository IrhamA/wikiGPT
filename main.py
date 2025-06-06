# Wrapper file that takes input (prompt) and returns output (answer)
import cohere
import wikipedia
from sentence_transformers import SentenceTransformer, util
from src.wikipedia_search import fetch_article, chunk_text
from src.embedder import embed_chunks, build_index
from src.retriever import retrieve
from src.answer import generate_answer

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize Cohere client and SentenceTransformer model
co = cohere.Client(API_KEY)
# Maps english sentences to 384-dimensional vectors
MODEL = SentenceTransformer('all-MiniLM-L6-v2') 


question = input("Ask wikipedia anything: ")
search_results = wikipedia.search(question, results = 10)
if not search_results:
    print("No results found.")
    exit()

# embed articles and user query
title_embeddings = MODEL.encode(search_results, convert_to_tensor=True)
query_embedding = MODEL.encode(question, convert_to_tensor=True)

# rank titles and obtain the top 3
cosine_scores = util.pytorch_cos_sim(query_embedding, title_embeddings)[0]
k = min(3, len(search_results))
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


