# write_db.py
from functools import lru_cache
from typing import Sequence, List
import numpy as np
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "clip_embeddings"

@lru_cache(maxsize=1)
def _get_client():
    """یک بار بساز و کش کن؛ بارهای بعدی همین را برگردان."""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

@lru_cache(maxsize=1)
def _get_collection():
    """کالکشن را هم کش کن تا هر بار ساخته نشود."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # برای CLIP
    )

def _to_python_float_list(vec: Sequence[float]) -> List[float]:
    """
    هر چی بهت بدن (np.array، list از np.float32، ...)،
    قطعی تبدیلش کن به list از float پایتونی.
    """
    arr = np.asarray(vec, dtype=np.float32)
    # tolist() روی np.float32 معمولاً float پایتونی می‌دهد،
    # ولی برای جلوگیری از edge-caseها، یک cast دیگر هم می‌کنیم:
    return arr.astype(float).tolist()

def add_embedding(id_str: str, embedding: Sequence[float]) -> None:
    """
    فقط id + embedding را ذخیره می‌کند.
    اگر upsert در سرورت پشتیبانی شود، از upsert استفاده می‌کنیم.
    در غیر این صورت حذف قبلی و add جدید.
    """
    col = _get_collection()
    emb_list = _to_python_float_list(embedding)

    # اگر upsert وجود دارد، همان بهتر
    if hasattr(col, "upsert"):
        col.upsert(ids=[id_str], embeddings=[emb_list])
        return

    # fallback: delete سپس add
    try:
        col.delete(ids=[id_str])
    except Exception:
        pass
    col.add(ids=[id_str], embeddings=[emb_list])
