#Goal: Build a FAISS index of Wikipedia article summaries from article titles
import wikipedia
import pickle
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Load your list of titles
with open("data/articles.txt") as f:
    titles = [line.strip() for line in f.readlines()]

summaries = []
valid_titles = []
#MAX_CHARS = 1000

for title in tqdm(titles, desc="Fetching & embedding summaries"):
    try:
        summary = wikipedia.summary(title)
        #summary = summary[:MAX_CHARS]
        summaries.append(summary)
        valid_titles.append(title)
    except:
        continue

# Embed all summaries
embeddings = MODEL.encode(summaries, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and titles
faiss.write_index(index, "wiki_longer_summary_index.faiss")
with open("wiki_longer_summary_titles.pkl", "wb") as f:
    pickle.dump(valid_titles, f)

print(f"Successfully indexed {len(valid_titles)} articles.")