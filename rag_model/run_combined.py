import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

image_encoder_path = current_dir.parent / "image_encoder"
text_encoder_path = current_dir.parent / "text_encoder"
sys.path.insert(0, str(image_encoder_path))
sys.path.insert(0, str(text_encoder_path))

print(f"🔧 Python path setup:")
print(f"   Current dir: {current_dir}")
print(f"   Image encoder: {image_encoder_path}")
print(f"   Text encoder: {text_encoder_path}")

try:
    print("\n📦 Importing modules...")
    
    import emb as image_emb
    print("✅ Image encoder imported")
    

    import emb_text as text_emb
    print("✅ Text encoder imported")
    
    import text_builder
    print("✅ Text builder imported")
    
    import json
    import numpy as np
    import chromadb
    print("✅ Other modules imported")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "combined_embeddings"
IMAGES_DIR = "../processed_images"
JSON_PATH = "../Foods-dataset/foods.normalized.json"

def get_combined_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def load_food_items():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    else:
        raise ValueError("Unsupported JSON structure")

def process_dish(item):
    dish_id = str(item['id'])
    title = item.get('title', '')
    province = (item.get('location', {}) or {}).get('province', '')
    
    print(f"\n🍽️  Processing dish {dish_id}: {title}")
    
    try:
        images_dir = Path(IMAGES_DIR)
        image_files = []
        
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            pattern = f"{dish_id}_{ext}"
            image_files.extend(images_dir.glob(pattern))
        
        image_files = list(set(image_files))
        
        print(f"  📸 Found {len(image_files)} images")
        
        image_embeddings = []
        for img_file in image_files:
            try:
                emb = image_emb.encode_image(str(img_file))
                image_embeddings.append(emb)
                print(f"    ✅ {img_file.name}")
            except Exception as e:
                print(f"    ❌ {img_file.name}: {e}")
        
        if not image_embeddings:
            print(f"  ⚠️  No valid images for dish {dish_id}")
            return False
        
        text = text_builder.build_text_from_item(item)
        if not text.strip():
            print(f"  ⚠️  Empty text for dish {dish_id}")
            return False
        
        text_embedding = text_emb.encode_text(text)
        print(f"  📝 Text encoded: {len(text)} chars")
        
        all_embeddings = image_embeddings + [text_embedding]
        combined = np.mean(all_embeddings, axis=0)
        combined = combined / np.linalg.norm(combined)
        combined = combined.astype(np.float32)
        
        col = get_combined_collection()
        metadata = {
            "type": "combined",
            "dish_id": dish_id,
            "title": title,
            "province": province,
            "num_images": len(image_embeddings),
            "has_text": True,
            "total_embeddings": len(all_embeddings)
        }
        
        col.upsert(
            ids=[dish_id],
            embeddings=[combined.astype(float).tolist()],
            metadatas=[metadata]
        )
        
        print(f"  ✅ Saved to database")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("🚀 Starting combined embedding generation...")
    
    if not Path(JSON_PATH).exists():
        print(f"❌ JSON file not found: {JSON_PATH}")
        return
    
    if not Path(IMAGES_DIR).exists():
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return
    
    print("📖 Loading food items...")
    items = load_food_items()
    print(f"Found {len(items)} items")
    
    success = 0
    total = len(items)
    
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{total}]", end="")
        if process_dish(item):
            success += 1
    
    print(f"\n🎉 Complete! {success}/{total} dishes processed successfully")

if __name__ == "__main__":
    main()
