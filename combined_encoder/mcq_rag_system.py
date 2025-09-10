import sys
import os
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

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
from rag_system import AdvancedRAGSystem

class MCQ_RAG_System:

    def __init__(self, llm_provider: str = "openai", api_key: str = None):
        self.rag_system = AdvancedRAGSystem(llm_provider, api_key)
        self.results = []
        
    def load_questions_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        questions = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    question_data = {
                        'question': row['سوال'].strip(),
                        'option1': row['گزینه یک'].strip(),
                        'option2': row['گزینه دو'].strip(),
                        'option3': row['گزینه سه '].strip(),
                        'option4': row['گزینه چهار'].strip(),
                        'correct_answer': row['پاسخ درست'].strip(),
                        'image_path': row['عکس'].strip() if row['عکس'] else None,
                        'question_type': row.get('نوع سوال', 'متنی').strip(),
                        'combined_text': row.get('ترکیب سوال و گزینه ها', '').strip()
                    }
                    questions.append(question_data)
                    
        except Exception as e:
            print(f"خطا در بارگذاری فایل CSV: {e}")
            return []
            
        return questions
    
    def format_question_for_rag(self, question_data: Dict[str, Any]) -> str:
        question = question_data['question']
        options = [
            f"گزینه 1: {question_data['option1']}",
            f"گزینه 2: {question_data['option2']}",
            f"گزینه 3: {question_data['option3']}",
            f"گزینه 4: {question_data['option4']}"
        ]
        
        formatted_question = f"{question}\n\n" + "\n".join(options)
        return formatted_question
    
    def determine_search_type(self, question_data: Dict[str, Any]) -> str:
        image_path = self.get_image_path(question_data)
        if image_path:
            if question_data['question'].startswith('این تصویر مربوط به کدام غذاست'):
                return 'image'
            else:
                return 'combined'
        else:
            return 'text'
    
    def get_image_path(self, question_data: Dict[str, Any]) -> Optional[str]:
        image_path = question_data.get('image_path')
        
        if image_path is None:
            return None
            
        image_path = str(image_path).strip()
        
        if not image_path or image_path == '' or image_path.lower() == 'null':
            return None
        
        if image_path.startswith('..\\sample_questions\\'):
            image_path = image_path.replace('..\\sample_questions\\', '')
        elif image_path.startswith('..\\'):
            image_path = image_path.replace('..\\', '')
        
        image_path = image_path.replace('\\', '/')
        
        if image_path.startswith('sample_questions/'):
            image_path = image_path.replace('sample_questions/', '')
        
        full_image_path = current_dir.parent / "sample_questions" / image_path
        
        if full_image_path.exists():
            try:
                from PIL import Image
                with Image.open(full_image_path) as img:
                    img.verify()
                return str(full_image_path)
            except Exception as e:
                print(f"❌ فایل تصویر خراب است: {full_image_path} - {e}")
                return None
        else:
            print(f"❌ تصویر یافت نشد: {full_image_path}")
            return None
    
    def extract_answer_from_response(self, response: str, question_data: Dict[str, Any]) -> str:
        response_lower = response.lower()
        
        options = [
            question_data['option1'],
            question_data['option2'], 
            question_data['option3'],
            question_data['option4']
        ]
        
        for i, option in enumerate(options, 1):
            if option.lower() in response_lower:
                return f"گزینه {i}"
        
        if 'گزینه 1' in response or 'گزینه یک' in response or 'گزینه1' in response:
            return 'گزینه 1'
        elif 'گزینه 2' in response or 'گزینه دو' in response or 'گزینه2' in response:
            return 'گزینه 2'
        elif 'گزینه 3' in response or 'گزینه سه' in response or 'گزینه3' in response:
            return 'گزینه 3'
        elif 'گزینه 4' in response or 'گزینه چهار' in response or 'گزینه4' in response:
            return 'گزینه 4'
        
        # جستجوی فقط شماره
        if '1' in response and 'گزینه' in response:
            return 'گزینه 1'
        elif '2' in response and 'گزینه' in response:
            return 'گزینه 2'
        elif '3' in response and 'گزینه' in response:
            return 'گزینه 3'
        elif '4' in response and 'گزینه' in response:
            return 'گزینه 4'
        
        return 'گزینه 1'
    
    def process_single_question(self, question_data: Dict[str, Any], question_id: int) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"سوال {question_id}: {question_data['question'][:100]}...")
        print(f"نوع سوال: {question_data['question_type']}")
        
        search_type = self.determine_search_type(question_data)
        print(f"نوع جستجو: {search_type}")
        
        formatted_question = self.format_question_for_rag(question_data)
        
        image_path = self.get_image_path(question_data)
        if image_path:
            print(f"تصویر: {Path(image_path).name}")
        else:
            print("تصویر: ندارد")
            if search_type in ['image', 'combined']:
                search_type = 'text'
                print(f"نوع جستجو تغییر یافت به: {search_type}")
        
        try:
            result = self.rag_system.generate_response(
                query=formatted_question,
                search_type=search_type,
                image_path=image_path,
                top_k=5
            )
            
            predicted_answer = self.extract_answer_from_response(
                result['response'], 
                question_data
            )
            
            correct_answer = question_data['correct_answer']
            if correct_answer.isdigit():
                correct_answer = f"گزینه {correct_answer}"
            elif correct_answer.startswith('گزینه'):
                pass
            else:
                if '1' in correct_answer or 'یک' in correct_answer:
                    correct_answer = "گزینه 1"
                elif '2' in correct_answer or 'دو' in correct_answer:
                    correct_answer = "گزینه 2"
                elif '3' in correct_answer or 'سه' in correct_answer:
                    correct_answer = "گزینه 3"
                elif '4' in correct_answer or 'چهار' in correct_answer:
                    correct_answer = "گزینه 4"
            
            is_correct = predicted_answer == correct_answer
            
            result_data = {
                'question_id': question_id,
                'question': question_data['question'],
                'question_type': question_data['question_type'],
                'search_type': search_type,
                'image_path': image_path,
                'options': {
                    'option1': question_data['option1'],
                    'option2': question_data['option2'],
                    'option3': question_data['option3'],
                    'option4': question_data['option4']
                },
                'correct_answer': question_data['correct_answer'],
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'rag_response': result['response'],
                'retrieved_docs': result['retrieved_docs'],
                'context': result['context']
            }
            
            print(f"پاسخ صحیح: {question_data['correct_answer']}")
            print(f"پاسخ پیش‌بینی شده: {predicted_answer}")
            print(f"نتیجه: {'✅ صحیح' if is_correct else '❌ اشتباه'}")
            
            return result_data
            
        except Exception as e:
            print(f"خطا در پردازش سوال {question_id}: {e}")
            return {
                'question_id': question_id,
                'question': question_data['question'],
                'error': str(e),
                'is_correct': False
            }
    
    def process_all_questions(self, csv_path: str, max_questions: int = None) -> List[Dict[str, Any]]:
        print("🤖 شروع پردازش سوالات چهارگزینه‌ای")
        print("=" * 80)
        
        questions = self.load_questions_from_csv(csv_path)
        if not questions:
            print("هیچ سوالی یافت نشد!")
            return []
        
        if max_questions:
            questions = questions[:max_questions]
        
        print(f"تعداد سوالات: {len(questions)}")
        
        results = []
        for i, question_data in enumerate(questions, 1):
            result = self.process_single_question(question_data, i)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def calculate_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.get('is_correct', False))
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        text_questions = [r for r in results if r.get('question_type') == 'متنی']
        image_questions = [r for r in results if r.get('question_type') == 'تصویری']
        combined_questions = [r for r in results if r.get('question_type') == 'ترکیبی']
        
        stats = {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'text_questions': {
                'count': len(text_questions),
                'correct': sum(1 for r in text_questions if r.get('is_correct', False)),
                'accuracy': sum(1 for r in text_questions if r.get('is_correct', False)) / len(text_questions) if text_questions else 0
            },
            'image_questions': {
                'count': len(image_questions),
                'correct': sum(1 for r in image_questions if r.get('is_correct', False)),
                'accuracy': sum(1 for r in image_questions if r.get('is_correct', False)) / len(image_questions) if image_questions else 0
            },
            'combined_questions': {
                'count': len(combined_questions),
                'correct': sum(1 for r in combined_questions if r.get('is_correct', False)),
                'accuracy': sum(1 for r in combined_questions if r.get('is_correct', False)) / len(combined_questions) if combined_questions else 0
            }
        }
        
        return stats
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None):
        if not output_path:
            output_path = f"mcq_results.json"
        
        clean_results = []
        for result in results:
            clean_result = {k: v for k, v in result.items() if k not in ['retrieved_docs', 'context']}
            clean_results.append(clean_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        print(f"نتایج در فایل {output_path} ذخیره شد.")
        
        stats = self.calculate_accuracy(results)
        stats_path = output_path.replace('.json', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"آمار در فایل {stats_path} ذخیره شد.")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        stats = self.calculate_accuracy(results)
        
        print("\n" + "="*80)
        print("📊 خلاصه نتایج")
        print("="*80)
        print(f"کل سوالات: {stats['total_questions']}")
        print(f"پاسخ‌های صحیح: {stats['correct_answers']}")
        print(f"دقت کلی: {stats['accuracy']:.2%}")
        print()
        print("تفکیک بر اساس نوع سوال:")
        print(f"  متنی: {stats['text_questions']['correct']}/{stats['text_questions']['count']} ({stats['text_questions']['accuracy']:.2%})")
        print(f"  تصویری: {stats['image_questions']['correct']}/{stats['image_questions']['count']} ({stats['image_questions']['accuracy']:.2%})")
        print(f"  ترکیبی: {stats['combined_questions']['correct']}/{stats['combined_questions']['count']} ({stats['combined_questions']['accuracy']:.2%})")

def main():
    print("🤖 سیستم RAG برای سوالات چهارگزینه‌ای")
    print("=" * 80)
    
    csv_path = current_dir.parent / "sample_questions" / "NLP_HW3_QUESTIONS.csv"
    
    if not csv_path.exists():
        print(f"فایل سوالات یافت نشد: {csv_path}")
        return
    
    mcq_rag = MCQ_RAG_System(
        llm_provider="openai", 
        api_key=""
    )
    
    results = mcq_rag.process_all_questions(str(csv_path), max_questions=200)
    
    mcq_rag.print_summary(results)
    
    mcq_rag.save_results(results)

if __name__ == "__main__":
    main()
