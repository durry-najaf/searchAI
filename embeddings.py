# backend/embeddings.py
import openai
import faiss
import numpy as np
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=openai.api_key)

dimension = 1536  # Embedding vector size
index = faiss.IndexFlatL2(dimension)
documents = []

def get_embedding(text: str):
    truncated = truncate_text(text)
    response = client.embeddings.create(
        input=[truncated],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def index_website(url, text):
    if not text:
        return
    embedding = get_embedding(text)
    embedding = embedding.reshape(1, -1)  # Ensure correct shape
    index.add(np.array(embedding).astype("float32"))  # Add to FAISS
    documents.append({"url": url, "content": text})
    print(f"Indexed: {url}")  # Debugging

def search_website(query):
    if not documents:
        return []
    query_embedding = get_embedding(query).reshape(1, -1)  # Ensure correct shape
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k=min(10, len(documents)))

    print(f"Search Query: {query}, Distances: {distances}, Indices: {indices}")  # Debugging

    results = []
    for i in indices[0]:
        if i < len(documents):
            results.append({"url": documents[i]["url"], "content": documents[i]["content"]})
    
    return results if results else [{"message": "No relevant results found."}]

def generate_ai_article(query):
    relevant_docs = search_website(query)
    if not relevant_docs:
        return "Sorry, I couldn't find enough information on that topic."

    combined_text = "\n\n".join(doc["content"][:1000] for doc in relevant_docs[:5])

    prompt = f"""
    You are a helpful assistant trained on internal documentation.
    Write an informative article about "{query}" based only on the following source material:

    {combined_text}

    Keep the tone professional and tailored to the insurance and brokerage industry. Avoid adding unrelated info.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful writing assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    

    return response.choices[0].message.content.strip()
