#!/usr/bin/env python3
"""
اسکریپت ساده برای اجرای سیستم میانگین embedding
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add parent directories to path
image_encoder_path = current_dir.parent / "image_encoder"
text_encoder_path = current_dir.parent / "text_encoder"
sys.path.insert(0, str(image_encoder_path))
sys.path.insert(0, str(text_encoder_path))

print(f"🔧 Python path setup:")
print(f"   Current dir: {current_dir}")
print(f"   Image encoder: {image_encoder_path}")
print(f"   Text encoder: {text_encoder_path}")

# Now import the modules
try:
    print("\n📦 Importing modules...")
    
    # Import image encoder
    import emb as image_emb
    print("✅ Image encoder imported")
    
    # Import text encoder  
    import emb_text as text_emb
    print("✅ Text encoder imported")
    
    # Import text builder
    import text_builder
    print("✅ Text builder imported")
    
    # Import other required modules
    import json
    import numpy as np
    import chromadb
    print("✅ Other modules imported")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configuration
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "combined_embeddings"
IMAGES_DIR = "../processed_images"
JSON_PATH = "../Foods-dataset/foods.normalized.json"

def get_combined_collection():
    """Get or create combined embeddings collection"""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def load_food_items():
    """Load food items from JSON file"""
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
    """Process a single dish"""
    dish_id = str(item['id'])
    title = item.get('title', '')
    province = (item.get('location', {}) or {}).get('province', '')
    
    print(f"\n🍽️  Processing dish {dish_id}: {title}")
    
    try:
        # Get image embeddings
        images_dir = Path(IMAGES_DIR)
        image_files = []
        
        # Find all images for this dish - pattern: {dish_id}_{seq}_{title}__{label}.jpg
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            pattern = f"{dish_id}_{ext}"
            image_files.extend(images_dir.glob(pattern))
        
        # Remove duplicates
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
        
        # Get text embedding
        text = text_builder.build_text_from_item(item)
        if not text.strip():
            print(f"  ⚠️  Empty text for dish {dish_id}")
            return False
        
        text_embedding = text_emb.encode_text(text)
        print(f"  📝 Text encoded: {len(text)} chars")
        
        # Compute combined embedding
        all_embeddings = image_embeddings + [text_embedding]
        combined = np.mean(all_embeddings, axis=0)
        combined = combined / np.linalg.norm(combined)
        combined = combined.astype(np.float32)
        
        # Save to database
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
    """Main function"""
    print("🚀 Starting combined embedding generation...")
    
    # Check if files exist
    if not Path(JSON_PATH).exists():
        print(f"❌ JSON file not found: {JSON_PATH}")
        return
    
    if not Path(IMAGES_DIR).exists():
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return
    
    # Load items
    print("📖 Loading food items...")
    items = load_food_items()
    print(f"Found {len(items)} items")
    
    # Process items
    success = 0
    total = len(items)
    
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{total}]", end="")
        if process_dish(item):
            success += 1
    
    print(f"\n🎉 Complete! {success}/{total} dishes processed successfully")

if __name__ == "__main__":
    main()
