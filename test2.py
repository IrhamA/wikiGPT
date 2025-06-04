from sentence_transformers import SentenceTransformer
import faiss
import wikipedia

API_KEY = "B8YtmWmwrCJ2QRf197v2YHIcdfx8IsJDouHu1UsM"

# 1. Load a transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast

# 2. Get full Wikipedia content (not just summary)
article = wikipedia.page("Black hole").content

# 3. Chunk the text
def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

chunks = chunk_text(article)

# 4. Generate embeddings for each chunk
embeddings = model.encode(chunks)

# 5. Index them using FAISS
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. When user asks a question, embed the query
question = "How are black holes formed?"
q_embedding = model.encode([question])

# 7. Search for top 3 most relevant chunks
D, I = index.search(q_embedding, k=3)

print(I)

# 8. Display matching chunks
print("Top relevant context chunks:")
for idx in I[0]:
    print("\n---\n", chunks[idx])


###############################################################################

import cohere

context = "\n".join([chunks[idx] for idx in I[0]])
question = "How are black holes formed?"

final_prompt = f"""Answer the question using only the information below:

Context: {context}

Question: {question}

Answer: """


co = cohere.Client(API_KEY)

response = co.chat(
    message = final_prompt,
    model="command-r",  # or whatever you have access to
    temperature=0.5
)


print("Answer:")
print(response.text.strip())