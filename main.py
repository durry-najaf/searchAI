# backend/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from scraper import scrape_website_parallel
from embeddings import get_embedding, search_website, index_website, generate_ai_article
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scrape and index a website (multi-page support with parallel processing)
@app.get("/index/")
def index(url: str, depth: int = 1):
    pages_scraped = scrape_website_parallel(url, depth)
    if not pages_scraped:
        return {"error": "Failed to scrape website or no valid pages found."}
    for page_url, text in pages_scraped.items():
        index_website(page_url, text)
    return {"message": "Website indexed successfully!", "pages_indexed": len(pages_scraped)}

# Search website content
@app.get("/search/")
def search(query: str = Query(..., min_length=1)):
    results = search_website(query)
    if not results:
        return {"message": "No results found."}
    return {"results": results}

# Generate AI article based on website content
@app.get("/generate-article/")
def generate_article(query: str = Query(..., min_length=1)):
    article = generate_ai_article(query)
    return {"article": article}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
