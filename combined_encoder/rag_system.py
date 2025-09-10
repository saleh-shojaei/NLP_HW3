import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import requests

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

image_encoder_path = current_dir.parent / "image_encoder"
text_encoder_path = current_dir.parent / "text_encoder"
sys.path.insert(0, str(image_encoder_path))
sys.path.insert(0, str(text_encoder_path))

import emb as image_emb
import emb_text as text_emb
import numpy as np
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "combined_embeddings"
IMAGES_DIR = "../processed_images"

class AdvancedRAGSystem:

    def __init__(self, llm_provider: str = "openai", api_key: str = None):
        self.collection = self._get_collection()
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.retrieved_docs = []
        
    def _get_collection(self):
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        return client.get_or_create_collection(name=COLLECTION_NAME)
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"Searching by text: '{query}'")
        
        query_emb = text_emb.encode_text(query)
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["distances", "metadatas"]
        )
        
        return self._format_results(results)
    
    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"🔍 Searching by image: {image_path}")
        
        query_emb = image_emb.encode_image(image_path)
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["distances", "metadatas"]
        )
        
        return self._format_results(results)
    
    def search_by_combined(self, image_path: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"Searching by combined: image='{image_path}', text='{query_text}'")
        
        image_embedding = image_emb.encode_image(image_path)
        text_embedding = text_emb.encode_text(query_text)
        combined = (image_embedding + text_embedding) / 2
        combined = combined / np.linalg.norm(combined)
        
        results = self.collection.query(
            query_embeddings=[combined.tolist()],
            n_results=top_k,
            include=["distances", "metadatas"]
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not ids:
            return []
        
        formatted_results = []
        for i, (dish_id, distance, metadata) in enumerate(zip(ids, distances, metadatas), 1):
            result = {
                "rank": i,
                "dish_id": dish_id,
                "distance": distance,
                "title": metadata.get("title", ""),
                "province": metadata.get("province", ""),
                "num_images": metadata.get("num_images", 0),
                "has_text": metadata.get("has_text", False),
                "total_embeddings": metadata.get("total_embeddings", 0)
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def retrieve_documents(self, query: str, search_type: str = "text", image_path: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if search_type == "text":
            self.retrieved_docs = self.search_by_text(query, top_k)
        elif search_type == "image":
            if not image_path:
                raise ValueError("Image path required for image search")
            self.retrieved_docs = self.search_by_image(image_path, top_k)
        elif search_type == "combined":
            if not image_path:
                raise ValueError("Image path required for combined search")
            self.retrieved_docs = self.search_by_combined(image_path, query, top_k)
        else:
            raise ValueError("Invalid search type. Use 'text', 'image', or 'combined'")
        
        return self.retrieved_docs
    
    def format_context_for_llm(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        
        context_parts.append("شما یک متخصص آشپزی ایرانی هستید. بر اساس اطلاعات زیر به سوال کاربر پاسخ دهید - دقت کن که سوالات چند گزینه‌ای است و بهترین پاسخ را انتخاب کن:")
        context_parts.append(f"\nسوال: {query}")
        context_parts.append(f"\nاطلاعات مرتبط:")
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"{i}. {doc['title']}")
            if doc['province']:
                context_parts.append(f"   استان: {doc['province']}")
            context_parts.append(f"   شباهت: {doc['distance']:.4f}")
        
        return "\n".join(context_parts)
    
    def call_openai_api(self, context: str, query: str) -> str:
        if not self.api_key:
            return "API key not provided for OpenAI"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5",
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling OpenAI API: {e}"
    
    def call_ollama_api(self, context: str, query: str, model: str = "llama2") -> str:
        """فراخوانی Ollama API (محلی)"""
        try:
            data = {
                "model": model,
                "prompt": f"{context}\n\nسوال: {query}",
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Ollama API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Ollama API: {e}"
    
    def generate_response(self, query: str, search_type: str = "text", image_path: str = None, top_k: int = 5) -> Dict[str, Any]:
        print(f"\n🤖 Generating response for query: '{query}'")
        
        retrieved_docs = self.retrieve_documents(query, search_type, image_path, top_k)
        
        if not retrieved_docs:
            return {
                "query": query,
                "retrieved_docs": [],
                "context": "",
                "response": "متأسفانه هیچ اطلاعات مرتبطی یافت نشد.",
                "search_type": search_type,
                "llm_provider": self.llm_provider
            }
        
        context = self.format_context_for_llm(query, retrieved_docs)


        if self.llm_provider == "openai":
            response = self.call_openai_api(context, query)
        elif self.llm_provider == "ollama":
            response = self.call_ollama_api(context, query)
        else:
            response = self._generate_simple_response(query, retrieved_docs)
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "response": response,
            "search_type": search_type,
            "llm_provider": self.llm_provider
        }
    
    def _generate_simple_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "متأسفانه هیچ اطلاعات مرتبطی یافت نشد."

        food_titles = [doc['title'] for doc in retrieved_docs]
        provinces = [doc['province'] for doc in retrieved_docs if doc['province']]

        response_parts = []
        response_parts.append(f"بر اساس جستجوی شما، {len(retrieved_docs)} غذای مرتبط یافت شد:")

        for i, doc in enumerate(retrieved_docs, 1):
            response_parts.append(f"{i}. {doc['title']}")
            if doc['province']:
                response_parts.append(f"   از استان {doc['province']}")

        if provinces:
            unique_provinces = list(set(provinces))
            response_parts.append(f"\nاین غذاها از استان‌های {', '.join(unique_provinces)} هستند.")

        return "\n".join(response_parts)

    def display_response(self, result: Dict[str, Any]):
        """نمایش پاسخ نهایی"""
        print(f"\n🎯 پاسخ نهایی ({result['llm_provider']}):")
        print("=" * 60)
        print(result['response'])
        
        print(f"\n📊 اطلاعات بازیابی شده:")
        print("-" * 40)
        for doc in result['retrieved_docs']:
            print(f"• {doc['title']} (شباهت: {doc['distance']:.4f})")

def main():
    """Main function"""
    print("🤖 Advanced RAG System")
    print("=" * 60)
    
    # Initialize RAG system
    # برای استفاده از OpenAI، API key خود را وارد کنید
    # rag = AdvancedRAGSystem(llm_provider="openai", api_key="your-api-key")

    # برای استفاده از Ollama (محلی)
    rag = AdvancedRAGSystem(llm_provider="openAI", api_key='tpsg-0V0Q8zout546HH63L16zWWxAR5b9MHj')
    
    # Example queries
    examples = [
        {
            "query": "غذای ایرانی با سبزی",
            "search_type": "text",
            "top_k": 3
        },
        {
            "query": "دسر شیرین",
            "search_type": "text", 
            "top_k": 3
        },
        {
            "query": "این غذا چیست؟",
            "search_type": "image",
            "image_path": "nan_berenji.jpeg",
            "top_k": 3
        }
    ]
    
    for example in examples:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {example['query']}")
        print(f"📊 Search Type: {example['search_type']}")
        print(f"🔢 Top K: {example['top_k']}")
        
        # Get image_path if it exists
        image_path = example.get('image_path', None)
        if image_path:
            print(f"📸 Image: {Path(image_path).name}")
        
        # Generate response
        result = rag.generate_response(
            query=example['query'],
            search_type=example['search_type'],
            top_k=example['top_k'],
            image_path=image_path,
        )
        
        # Display response
        rag.display_response(result)
        
        print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
