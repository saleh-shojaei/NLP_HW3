# search_text_by_query.py
import sys
import numpy as np
import chromadb
from emb_text import encode_text  # همان انکودر متنی شما (multilingual CLIP)
from typing import List

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "text_embeddings"  # کالکشن متن

def fetch_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def search_text(query: str, top_k: int = 10):
    col = fetch_collection()
    q_emb = encode_text(query)  # np.ndarray (d,)
    # Chroma API برای query_embeddings یک لیست از لیست‌ها می‌خواهد
    res = col.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
    ids: List[str] = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]  # فاصله کساین: کوچکتر=بهتر
    metas = res.get("metadatas", [[]])[0] if "metadatas" in res else [{} for _ in ids]

    if not ids:
        print("No results.")
        return

    print(f"Top-{len(ids)} results for query: {query!r}")
    # مرتب‌سازی دستی اگر لازم شد (بعضی نسخه‌ها مرتب‌شده برمی‌گردانند)
    results = list(zip(ids, dists, metas))
    results.sort(key=lambda x: x[1])  # distance asc
    for rank, (vid, dist, meta) in enumerate(results, start=1):
        title = (meta or {}).get("title", "")
        province = (meta or {}).get("province", "")
        dish_id = (meta or {}).get("dish_id", "")
        print(f"{rank:>2}. id={vid}  distance={dist:.6f}  dish_id={dish_id}  title={title}  province={province}")

def usage():
    print(
        "Usage:\n"
        "  python search_text_by_query.py <query text> [top_k]\n\n"
        "Examples:\n"
        "  python search_text_by_query.py \"غذای ایرانی با سبزی و لوبیا\" 5\n"
        "  python search_text_by_query.py \"دسر با پسته\""
    )

def main():
    if len(sys.argv) < 2:
        usage()
        return
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 10
    search_text(query, top_k)

if __name__ == "__main__":
    main()
