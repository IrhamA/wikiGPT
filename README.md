This project serves as a generative AI model that allows users to query data exclusively from wikipedia sources

Given a query find all relevant articles relating to that query.

Initial problems:
    As opposed to using AI to directly solve the problem, I use it as a tool to help steer me towards the correct answer

Chunk & Embed Wikipedia Pages (Vector Search)
Instead of just looking at summaries:
    Break pages into chunks (100–300 words)
    Embed each chunk using sentence-transformers or Cohere
    When a user asks a question, embed that too
    Find the top N relevant chunks with cosine similarity

Example Prompts:
 How do flashbangs work?
 X vs Y
 ...
