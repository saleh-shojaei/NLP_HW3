#!/usr/bin/env python3
"""
اسکریپت ساده برای جستجو در میانگین embedding ها
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

# Import modules
import emb as image_emb
import emb_text as text_emb
import numpy as np
import chromadb

# Configuration
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "combined_embeddings"
IMAGES_DIR = "../processed_images"

def get_collection():
    """Get collection"""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def search_text(query, top_k=5):
    """Search by text"""
    print(f"🔍 Searching by text: '{query}'")
    
    # Encode query
    query_emb = text_emb.encode_text(query)
    
    # Search
    col = get_collection()
    results = col.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k,
        include=["distances", "metadatas"]
    )
    
    display_results(results, "text")

def search_image(image_path, top_k=5):
    """Search by image"""
    print(f"🔍 Searching by image: {image_path}")
    
    # Encode query
    query_emb = image_emb.encode_image(image_path)
    
    # Search
    col = get_collection()
    results = col.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k,
        include=["distances", "metadatas"]
    )
    
    display_results(results, "image")

def search_combined(image_path, query_text, top_k=5):
    """Search by both image and text"""
    print(f"🔍 Searching by combined: image='{image_path}', text='{query_text}'")
    
    # Encode both
    image_embedding = image_emb.encode_image(image_path)
    text_embedding = text_emb.encode_text(query_text)
    
    # Average
    combined = (image_embedding + text_embedding) / 2
    combined = combined / np.linalg.norm(combined)
    
    # Search
    col = get_collection()
    results = col.query(
        query_embeddings=[combined.tolist()],
        n_results=top_k,
        include=["distances", "metadatas"]
    )
    
    display_results(results, "combined")

def display_results(results, search_type):
    """Display results"""
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    if not ids:
        print("❌ No results found")
        return
    
    print(f"\n📊 Top {len(ids)} results for {search_type} search:")
    print("=" * 60)
    
    for i, (dish_id, distance, metadata) in enumerate(zip(ids, distances, metadatas), 1):
        meta = metadata or {}
        print(f"{i:2d}. 🍽️  {dish_id}: {meta.get('title', 'N/A')}")
        print(f"    🗺️  {meta.get('province', 'N/A')} | 📏 {distance:.4f}")
        print(f"    📸 {meta.get('num_images', 0)} images | 📄 {meta.get('has_text', False)}")
        print()

def show_info():
    """Show collection info"""
    col = get_collection()
    count = col.count()
    print(f"📊 Collection '{COLLECTION_NAME}': {count} items")
    
    if count > 0:
        sample = col.get(limit=3, include=["metadatas"])
        print("\n📋 Sample items:")
        for dish_id, metadata in zip(sample["ids"], sample["metadatas"]):
            meta = metadata or {}
            print(f"  - {dish_id}: {meta.get('title', 'N/A')}")

def usage():
    """Show usage"""
    print("Usage:")
    print("  python run_search.py text <query> [top_k]")
    print("  python run_search.py image <image_path> [top_k]")
    print("  python run_search.py combined <image_path> <query> [top_k]")
    print("  python run_search.py info")
    print()
    print("Examples:")
    print("  python run_search.py text 'غذای ایرانی' 5")
    print("  python run_search.py image ../processed_images/6_1_قیقاناخ__2.jpg 5")
    print("  python run_search.py combined ../processed_images/6_1_قیقاناخ__2.jpg 'دسر' 5")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        usage()
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "text":
            if len(sys.argv) < 3:
                print("❌ Query text required")
                return
            query = sys.argv[2]
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 5
            search_text(query, top_k)
            
        elif command == "image":
            if len(sys.argv) < 3:
                print("❌ Image path required")
                return
            image_path = sys.argv[2]
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 5
            search_image(image_path, top_k)
            
        elif command == "combined":
            if len(sys.argv) < 4:
                print("❌ Both image path and query text required")
                return
            image_path = sys.argv[2]
            query = sys.argv[3]
            top_k = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 5
            search_combined(image_path, query, top_k)
            
        elif command == "info":
            show_info()
            
        else:
            print(f"❌ Unknown command: {command}")
            usage()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
