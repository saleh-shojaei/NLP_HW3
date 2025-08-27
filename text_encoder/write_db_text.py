# write_db_text.py
from functools import lru_cache
from typing import Sequence, List, Optional, Dict
import numpy as np
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "text_embeddings"


@lru_cache(maxsize=1)
def _get_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@lru_cache(maxsize=1)
def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # مناسب برای بردارهای نرمال‌شده (cosine)
    )


def _to_python_float_list(vec: Sequence[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.astype(float).tolist()


def add_embedding(id_str: str, embedding: Sequence[float], metadata: Optional[Dict] = None) -> None:
    """
    یک embedding متنی را به کالکشن ذخیره می‌کند.
    اگر سرور upsert داشته باشد، از upsert استفاده می‌کنیم؛ در غیر این صورت delete+add.
    """
    col = _get_collection()
    emb_list = _to_python_float_list(embedding)
    meta_list = [metadata or {}]

    if hasattr(col, "upsert"):
        col.upsert(ids=[id_str], embeddings=[emb_list], metadatas=meta_list)
        return

    try:
        col.delete(ids=[id_str])
    except Exception:
        pass
    col.add(ids=[id_str], embeddings=[emb_list], metadatas=meta_list)
