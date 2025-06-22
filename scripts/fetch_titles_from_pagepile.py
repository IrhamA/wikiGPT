# Goal: Using pagepile ID's, fetch titles and store the content in data/articles.txt

import requests
from pathlib import Path

def fetch_pagepile_titles(pile_id):
    url = f"https://pagepile.toolforge.org/api.php?action=get_data&id={pile_id}&format=json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("pages", [])

def qid_to_enwiki_title(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        data = requests.get(url).json()
        return data["entities"][qid]["sitelinks"]["enwiki"]["title"]
    except Exception as e:
        print(f"Failed to convert QID {qid}: {e}")
        return None

if __name__ == "__main__":
    """
    Fetch titles from a specific pagepile ID and convert QIDs to English Wikipedia titles.
    The titles are saved to data/articles.txt.
    """

    pile_id = 68383 # Choose your own pagepile ID here for a specific set of articles
    qids = fetch_pagepile_titles(pile_id)

    titles = []
    for qid in qids:
        title = qid_to_enwiki_title(qid)
        if title:
            titles.append(title)

    print(f"\nConverted {len(titles)} QIDs to titles")

    with open("data/articles.txt", "w", encoding="utf-8") as f:
        for title in titles:
            f.write(title + "\n")