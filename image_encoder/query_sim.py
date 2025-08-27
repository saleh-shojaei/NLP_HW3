# query_sim.py
import sys
import numpy as np
import chromadb
from emb import encode_image

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "clip_embeddings"

def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Query by image: {img_path}")
        emb = encode_image(img_path)
    else:
        print("No image path provided, using random normalized vector for demo")
        emb = np.random.rand(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

    # اتصال به سرور Chroma
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    # توجه: include نباید 'ids' داشته باشد
    results = col.query(
        query_embeddings=[emb.astype(np.float32).astype(float).tolist()],
        n_results=5,
        include=["distances"]  # 'ids' خودکار برمی‌گرده
    )

    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]
    if not ids:
        print("No results.")
        return

    print("\nTop results:")
    for rid, dist in zip(ids, dists):
        print(f"  id={rid}, distance={dist:.4f}")

if __name__ == "__main__":
    main()
