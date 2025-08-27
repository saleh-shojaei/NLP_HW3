# get_by_id_or_family.py
import sys
import re
import numpy as np
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "clip_embeddings"

def fetch_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def fetch_exact(col, vec_id: str):
    """برگشت embedding برای یک id دقیق"""
    res = col.get(ids=[vec_id], include=["embeddings"])
    ids = res.get("ids", [])
    if not ids:
        print("Not found.")
        return
    emb = np.array(res["embeddings"][0], dtype=np.float32)
    print(f"Found id: {ids[0]}")
    print(f"Embedding shape: {emb.shape}")
    print("First 8 values:", [float(f"{v:.6f}") for v in emb[:8]])

def iter_all_ids(col, page_size: int = 1000):
    """
    همه‌ی idها را صفحه‌به‌صفحه می‌آورد.
    نکته: where={} نفرست! (باعث خطا می‌شود)
    """
    offset = 0
    while True:
        res = col.get(
            limit=page_size,
            offset=offset,
            include=[]  # فقط ids
        )
        ids = res.get("ids", [])
        if not ids:
            break
        for _id in ids:
            yield _id
        offset += len(ids)

def fetch_family(col, dish_id: str):
    """
    همه‌ی اعضای یک خانواده: id هایی که با '<dish_id>_' شروع می‌شوند.
    چون متادیتا نداریم، همه‌ی ids را می‌خوانیم و با پیشوند فیلتر می‌کنیم.
    """
    prefix = f"{dish_id}_"
    fam_ids = [vid for vid in iter_all_ids(col) if vid.startswith(prefix)]
    if not fam_ids:
        print(f"No members found for dish_id='{dish_id}'")
        return

    print(f"Found {len(fam_ids)} member(s): {', '.join(fam_ids)}")

    # در صورت زیاد بودن، تکه‌تکه بگیریم
    B = 512
    all_ids = []
    all_embs = []
    for i in range(0, len(fam_ids), B):
        chunk = fam_ids[i:i+B]
        res = col.get(ids=chunk, include=["embeddings"])
        all_ids.extend(res.get("ids", []))
        all_embs.extend(res.get("embeddings", []))

    print("\nMembers and shapes:")
    for vid, v in zip(all_ids, all_embs):
        emb = np.array(v, dtype=np.float32)
        print(f"  - {vid}: shape={emb.shape}, first4={[float(f'{x:.6f}') for x in emb[:4]]}")

def usage():
    print(
        "Usage:\n"
        "  python get_by_id_or_family.py exact <id>\n"
        "  python get_by_id_or_family.py family <dish_id>\n\n"
        "Examples:\n"
        "  python get_by_id_or_family.py exact 6_4\n"
        "  python get_by_id_or_family.py family 6\n"
    )

def main():
    if len(sys.argv) < 3:
        usage()
        return

    mode = sys.argv[1].lower()
    key = sys.argv[2]

    col = fetch_collection()

    if mode == "exact":
        fetch_exact(col, key)
    elif mode == "family":
        if not re.match(r"^[^_]+$", key):
            print("dish_id should be the part before first underscore, e.g., '6'")
            return
        fetch_family(col, key)
    else:
        usage()

if __name__ == "__main__":
    main()
