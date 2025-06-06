# Goal: Fetch Wikipedia articles and break them into chunks.
import wikipedia

def fetch_article(title: str) -> str:
    """
    Fetch the content of a Wikipedia article by its title.
    """
    return wikipedia.page(title).content

def chunk_text(text: str, chunk_size: int = 1000, overlap=50) -> list[str]:
    """
    Breaks down a long string of text into smaller overlapping chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks