# WikiGPT
A minimal question-answering tool that uses Wikipedia content and an LLM to answer natural language queries.

## Features
- Semantic search over Wikipedia article titles
- Chunk-based retrieval using sentence embeddings and FAISS
- Context-aware answer generation with Cohere's LLM
- Modular, extensible architecture

## How It Works
1. User chooses to ask a question or explore related topics

2. For questions:
- Wikipedia titles are ranked by semantic similarity to the query
- Top articles are fetched, chunked, embedded, and indexed
- Relevant chunks are retrieved via FAISS
- Cohere LLM generates an answer using these chunks

3. For topics:
- Precomputed summary embeddings are searched for related concepts 

## Technologies
- Python, Wikipedia API, sentence-transformers, FAISS, Cohere
- dotenv for config, modular Python for each stage

## Usage
```bash
pip install -r requirements.txt
python3 scripts/fetch_titles_from_pagepile.py
python3 scripts/build_summary_index.py
python3 main.py