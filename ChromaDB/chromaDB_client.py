import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="localhost",
    port=8000
)

collection = client.get_or_create_collection(name="clip_embeddings")

collection.add(
    ids=["doc1"],
    embeddings=[[0.1] * 512],
    metadatas=[{"type": "text", "content": "a cute cat on a sofa"}]
)

collection.add(
    ids=["img1"],
    embeddings=[[0.2] * 512],
    metadatas=[{"type": "image", "path": "/images/cat.jpg"}]
)

results = collection.query(
    query_embeddings=[[0.1] * 512],
    n_results=2
)

print(results)
