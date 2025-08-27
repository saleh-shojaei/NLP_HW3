# emb_text.py
from functools import lru_cache
from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer

# متن‌ها را به فضای CLIP ViT-B/32 می‌برد (چندزبانه)
MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

@lru_cache(maxsize=1)
def _load_model_cpu():
    device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model

def encode_text(text: Union[str, list[str]]) -> np.ndarray:
    """
    متن (یا لیستی از متن‌ها) را به بردار 512بُعدی L2-normalized در فضای CLIP انکود می‌کند.
    خروجی: np.ndarray با dtype=float32
    """
    model = _load_model_cpu()
    if isinstance(text, str):
        vec = model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)[0]  # (512,)
    vecs = model.encode(text, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)        # (N, 512)
