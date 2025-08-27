# emb.py
from functools import lru_cache
from typing import Union
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

@lru_cache(maxsize=1)
def _load_model_cpu():
    device = "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    return model, processor, device

def encode_image(image_path: Union[str, bytes]) -> np.ndarray:
    """
    یک تصویر را با CLIP ViT-B/32 روی CPU انکد می‌کند.
    خروجی: بردار float32 با طول 512 و L2-normalized.
    """
    model, processor, device = _load_model_cpu()
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=[img], return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)           # [1, 512]
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        emb = feats.cpu().numpy().astype(np.float32)[0]      # (512,)
    return emb
