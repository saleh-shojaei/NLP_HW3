#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ارزیاب چندگزینه‌ای متن+عکس با مدل‌های شبیه LLaVA (لوکال و بدون دانلود خودکار)
- ورودی: ../sample_questions/NLP_HW3_QUESTIONS.csv
- خروجی: results_multimodal_mcq.csv (در کنار اسکریپت)
- مدل: GGUF محلی + فایل mmproj متناظر (برای ورودی عکس)

اجرای نمونه:
python llava_phi2_3b_ggm.py \
  --model ./models/ggml-model-Q4_K_M.gguf \
  --mmproj ./models/mmproj-model-f16.gguf \
  --csv ../sample_questions/NLP_HW3_QUESTIONS.csv \
  --images-root ..
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava15ChatHandler


def normalize_col(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().replace("\u200c", "").replace("\ufeff", "")


def find_persian_answer_number(text: str):
    """
    استخراج یکی از اعداد 1..4 از خروجی مدل.
    پشتیبانی از رقم/واژهٔ فارسی و حروف A-D انگلیسی.
    """
    if not text:
        return None

    t = str(text).strip()

    # 1) مستقیم: رقم 1..4
    m = re.search(r"\b([1-4])\b", t)
    if m:
        return int(m.group(1))

    # 2) واژگان/ارقام فارسی
    mapping = {
        "یک": 1, "۱": 1, "اول": 1,
        "دو": 2, "۲": 2, "دوم": 2,
        "سه": 3, "۳": 3, "سوم": 3,
        "چهار": 4, "۴": 4, "چهارم": 4,
    }
    for k, v in mapping.items():
        if re.search(fr"\b{k}\b", t):
            return v

    # 3) حروف A-D
    letters = {"A": 1, "B": 2, "C": 3, "D": 4}
    m = re.search(r"\b([A-D])\b", t, re.IGNORECASE)
    if m:
        return letters[m.group(1).upper()]

    return None


def build_user_content(full_question_text: str, image_path: Path | None):
    """
    ساخت content سازگار با فرمت LLaVA-1.5 (llama-cpp-python):
    - همیشه list از پارت‌ها: متن + (اختیاری) تصویر
    - تصویر با schema image_url و file://
    """
    parts = [{"type": "text", "text": full_question_text}]
    if image_path is not None:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"file://{image_path.as_posix()}"}
        })
    return parts


def extract_choice_from_toplogprobs(top_logprobs_list):
    """
    از top_logprobs توکن اول، logprob چهار رقم «1..4» و «۱..۴» را می‌گیرد
    و بیشینه را برمی‌گرداند.
    """
    if not top_logprobs_list:
        return None

    # tokens we consider
    ascii_digits = ["1", "2", "3", "4"]
    fa_digits = ["۱", "۲", "۳", "۴"]

    best_token = None
    best_lp = -1e30

    for item in top_logprobs_list:
        tok = item.get("token", "")
        lp = item.get("logprob", -1e30)
        # فقط توکن‌های تکیِ موردنظر
        if tok in ascii_digits:
            idx = ascii_digits.index(tok) + 1
        elif tok in fa_digits:
            idx = fa_digits.index(tok) + 1
        else:
            continue

        if lp > best_lp:
            best_lp = lp
            best_token = idx

    return best_token


def main():
    parser = argparse.ArgumentParser(description="ارزیاب MCQ چندوجهی با LLaVA (لوکال)")
    parser.add_argument("--model", required=True, help="مسیر فایل GGUF مدل (مثلاً ggml-model-Q4_K_M.gguf)")
    parser.add_argument("--mmproj", required=True, help="مسیر فایل mmproj متناظر (مثلاً mmproj-model-f16.gguf)")
    parser.add_argument("--csv", default="../sample_questions/NLP_HW3_QUESTIONS.csv",
                        help="مسیر CSV سؤالات")
    parser.add_argument("--images-root", default=".", help="ریشهٔ مسیرهای نسبیِ ستون «عکس»")
    parser.add_argument("--out", default="results_multimodal_llava_phi2_3b.csv",
                        help="نام فایل خروجی CSV")
    parser.add_argument("--ctx", type=int, default=4096, help="Context window (برای تصویر بهتر ≥4096)")
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                        help="برای GPU: -1 یا تعدادی از لایه‌ها (CUDA/Metal فعال باشد)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=1, help="خروجی فقط 1 توکن (1..4)")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    mmproj_path = Path(args.mmproj).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()

    if not model_path.exists():
        print(f"[خطا] فایل مدل پیدا نشد: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not mmproj_path.exists():
        print(f"[خطا] فایل mmproj پیدا نشد: {mmproj_path}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"[خطا] فایل CSV پیدا نشد: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # بارگذاری مدل با هندلر LLaVA
    chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
    llm = Llama(
        model_path=str(model_path),
        chat_handler=chat_handler,
        n_ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        logits_all=True,
        seed=args.seed,
        verbose=False,
    )

    # --- Grammar: فقط اجازهٔ 1 یا 2 یا 3 یا 4 (و حداکثر یک \n) ---
    grammar = LlamaGrammar.from_string(r"""
root ::= choice newline?
choice ::= "1" | "2" | "3" | "4" | "۱" | "۲" | "۳" | "۴"
newline ::= "\n"
""")

    # خواندن CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", engine="python")
    except Exception as e:
        print(f"[خطا] در خواندن CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # نرمال‌سازی نام ستون‌ها
    df.columns = [normalize_col(c) for c in df.columns]

    # نگاشت نام ستون‌های ممکن
    possible_cols = {
        "question": ["سوال", "پرسش"],
        "opt1": ["گزینه یک", "گزینه1", "گزینه۱", "گزینه اول"],
        "opt2": ["گزینه دو", "گزینه2", "گزینه۲"],
        "opt3": ["گزینه سه", "گزینه3", "گزینه۳"],
        "opt4": ["گزینه چهار", "گزینه4", "گزینه۴"],
        "answer": ["پاسخ درست", "جواب درست", "کلید"],
        "image": ["عکس", "تصویر", "مسیر عکس"],
        "combined": ["ترکیب سوال و گزینه ها", "سوال کامل", "متن کامل سوال"]
    }

    def pick(col_keys):
        for k in col_keys:
            if k in df.columns:
                return k
        return None

    COL_Q = pick(possible_cols["question"])
    COL_O1 = pick(possible_cols["opt1"])
    COL_O2 = pick(possible_cols["opt2"])
    COL_O3 = pick(possible_cols["opt3"])
    COL_O4 = pick(possible_cols["opt4"])
    COL_A = pick(possible_cols["answer"])
    COL_IMG = pick(possible_cols["image"])
    COL_COMBINED = pick(possible_cols["combined"])

    # few-shot کوچک برای کاهش سوگیری «همیشه 1»
    FEWSHOT = [
        {
            "role": "system",
            "content": ("شما یک دستیار ارزیابی چندگزینه‌ای هستید. "
                        "فقط شمارهٔ گزینهٔ درست را (یکی از 1،2،3،4) برمی‌گردانید.")
        },
        # مثال 1: پاسخ 2
        {"role": "user", "content": [{"type": "text", "text":
            "سوال: کوچکترین عدد بین 1, 3 و 7 و 5 کدام است؟\nگزینه 1: 1\nگزینه 2: 3\nگزینه 3: 5\nگزینه 4: 7"}]},
        {"role": "assistant", "content": "1"},
        # مثال 1: پاسخ 2
        {"role": "user", "content": [{"type": "text", "text":
            "سوال: بزرگترین عدد بین 3 و 7 و 5 کدام است؟\nگزینه 1: 3\nگزینه 2: 7\nگزینه 3: 5\nگزینه 4: 4"}]},
        {"role": "assistant", "content": "2"},
        # مثال 2: پاسخ 4
        {"role": "user", "content": [{"type": "text", "text":
            "سوال: حاصل 2×2 چند است؟\nگزینه 1: 3\nگزینه 2: 3.5\nگزینه 3: 5\nگزینه 4: 4"}]},
        {"role": "assistant", "content": "4"},
        # مثال 3: پاسخ 3
        {"role": "user", "content": [{"type": "text", "text":
            "سوال: رنگ آسمان در روز صاف؟\nگزینه 1: قرمز\nگزینه 2: سبز\nگزینه 3: آبی\nگزینه 4: زرد"}]},
        {"role": "assistant", "content": "3"},
        # حالا می‌رویم سراغ سؤال واقعی
    ]

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        try:
            # متن کامل سؤال: ترجیح با ستون «ترکیب سوال و گزینه ها»
            if COL_COMBINED and not pd.isna(row[COL_COMBINED]) and str(row[COL_COMBINED]).strip():
                full_text = normalize_col(row[COL_COMBINED])
            else:
                parts = []
                if COL_Q and not pd.isna(row[COL_Q]):
                    parts.append(normalize_col(row[COL_Q]))
                for i, col in enumerate([COL_O1, COL_O2, COL_O3, COL_O4], start=1):
                    if col and not pd.isna(row[col]):
                        parts.append(f"گزینه {i}: {normalize_col(row[col])}")
                full_text = "\n".join([p for p in parts if p])

            # تصویر (اختیاری)
            image_path = None
            if COL_IMG and not pd.isna(row[COL_IMG]):
                raw_img = str(row[COL_IMG]).strip()
                if raw_img:
                    candidate = (images_root / raw_img).resolve()
                    if candidate.exists():
                        try:
                            _ = Image.open(candidate).convert("RGB")
                            image_path = candidate
                        except Exception:
                            image_path = candidate

            # پیام‌ها: FEWSHOT + نمونهٔ فعلی
            messages = list(FEWSHOT) + [
                {"role": "user", "content": build_user_content(full_text, image_path)}
            ]

            # فراخوانی با Grammar + logprobs
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=args.max_tokens,  # 1
                top_k=4,
                top_p=1.0,
                grammar=grammar,
                logprobs=True,
                top_logprobs=10,
            )

            # خروجی نمونه‌برداری‌شده (ممکن است همیشه '1' شود)
            sampled = resp["choices"][0]["message"]["content"].strip()

            # استخراج توکن‌های کاندید با logprob
            # ساختار: resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"] -> list of dicts
            top_lp_list = None
            try:
                top_lp_list = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except Exception:
                top_lp_list = None

            prob_choice = extract_choice_from_toplogprobs(top_lp_list)
            # اگر به هر دلیل logprobs در دسترس نبود، همان sampled را پارس کن
            if prob_choice is None:
                model_choice = find_persian_answer_number(sampled)
            else:
                model_choice = prob_choice

            # کلید صحیح
            correct = None
            if COL_A and not pd.isna(row[COL_A]):
                correct = find_persian_answer_number(str(row[COL_A]))

            is_correct = (model_choice == correct) if (model_choice is not None and correct is not None) else None

            results.append({
                "index": idx,
                "سوال کامل": full_text,
                "گزینه انتخابی مدل (بر اساس logprob)": model_choice,
                "خروجی خام مدل (نمونه)": sampled,
                "پاسخ درست": correct,
                "عکس داشت؟": bool(image_path),
                "درست/نادرست": is_correct
            })

            print(f"[{idx + 1}/{total}] انتخاب: {model_choice} | نمونه: {sampled} | کلید: {correct} | "
                  f"{'✓' if is_correct else '✗' if is_correct is not None else '-'}")

        except KeyboardInterrupt:
            print("\n[توقف توسط کاربر]")
            break
        except Exception as e:
            print(f"[هشدار] خطا روی ردیف {idx}: {e}", file=sys.stderr)
            results.append({
                "index": idx,
                "سوال کامل": full_text if 'full_text' in locals() else "",
                "گزینه انتخابی مدل (بر اساس logprob)": None,
                "خروجی خام مدل (نمونه)": f"[ERROR] {e}",
                "پاسخ درست": None,
                "عکس داشت؟": False,
                "درست/نادرست": None
            })

    out_path = Path(args.out).resolve()
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"\n✅ ذخیره شد: {out_path}")


if __name__ == "__main__":
    main()
