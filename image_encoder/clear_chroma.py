# clear_chroma.py
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "clip_embeddings"

def main():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Delete failed (maybe not exists): {e}")

    # دوباره یک کالکشن خالی بسازیم تا آماده باشد
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Re-created empty collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
