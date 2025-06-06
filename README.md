# WikiGPT
A minimal question-answering tool that uses Wikipedia content and an LLM to answer natural language queries.

## Features
- Semantic search over Wikipedia article titles
- Chunk-based retrieval using sentence embeddings and FAISS
- Context-aware answer generation with Cohere's LLM
- Modular, extensible architecture

## How It Works
1. User asks a question.
2. Wikipedia titles are retrieved based on semantic similarity.
3. Articles are fetched, chunked, and embedded.
4. Top relevant chunks are selected via FAISS.
5. A language model generates an answer using the chunks.

## Technologies
- Python, Wikipedia API, sentence-transformers, FAISS, Cohere
- dotenv for config, modular Python for each stage

## Usage
```bash
pip install -r requirements.txt
python3 main.py