# -*- coding: utf-8 -*-
"""
Evaluate multimodal MCQs (text + optional image) using Qwen2-VL.

Input CSV: ../sample_questions/NLP_HW3_QUESTIONS.csv
  Columns (Persian): سوال, گزینه یک, گزینه دو, گزینه سه, گزینه چهار, پاسخ درست, عکس, ترکیب سوال و گزینه ها, ...
We ONLY use the fully-composed prompt from 'ترکیب سوال و گزینه ها'.
If 'عکس' has a valid path, we include that image.

Output CSV: predictions_qwen2vl.csv
  Columns: row_id, question_used, image_used, option1..4, gold_index, pred_index, pred_label, is_correct, model_output_raw

Model (default): Qwen/Qwen2-VL-2B-Instruct
You can override with --model <hf_repo_id>

Usage:
    python eval_multimodal_mcq.py
    python eval_multimodal_mcq.py --model Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration

# ---------------------- Config ----------------------
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
INPUT_CSV = Path("../sample_questions/NLP_HW3_QUESTIONS.csv")
OUTPUT_CSV = Path("predictions_qwen2vl.csv")
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # برای پاسخ قطعی (فقط 1..4)
# ---------------------------------------------------

def smart_open_image(path_str: str) -> Optional[Image.Image]:
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    p = Path(path_str.strip().strip('"').strip("'"))
    # اگر مسیر نسبی است، نسبت به محل CSV ورودی هم امتحان کنیم
    if not p.exists():
        # تلاش دوم: اگر ورودی نسبی باشد نسبت به پوشه CSV
        p2 = (INPUT_CSV.parent / p).resolve()
        if p2.exists():
            p = p2
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

def extract_digit_1_to_4(text: str) -> Optional[int]:
    """
    از خروجی مدل، اولین عدد 1..4 (لاتین یا فارسی) را بیرون می‌کشد.
    """
    if not text:
        return None
    # اعداد فارسی به لاتین
    fa_to_en = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    t = text.translate(fa_to_en)
    m = re.search(r"\b([1-4])\b", t)
    if m:
        return int(m.group(1))
    # گاهی مدل می‌نویسد: "گزینه 3" یا "Answer: 2"
    m = re.search(r"(?:گزینه|option|answer|ans)\D*([1-4])", t, re.I)
    if m:
        return int(m.group(1))
    return None

def pick_by_fuzzy_match(model_text: str, options: List[str]) -> Optional[int]:
    """
    اگر مدل به‌جای عدد، نام گزینه را برگرداند، با تطبیق ساده نزدیک‌ترین گزینه را انتخاب می‌کنیم.
    """
    from difflib import SequenceMatcher
    if not model_text:
        return None
    scores = []
    for i, opt in enumerate(options, start=1):
        ratio = SequenceMatcher(None, model_text.strip(), opt.strip()).ratio()
        scores.append((ratio, i))
    scores.sort(reverse=True)
    best_ratio, best_idx = scores[0]
    return best_idx if best_ratio >= 0.4 else None

def build_messages(full_prompt_text: str, img: Optional[Image.Image]) -> Tuple[list, list]:
    """
    پیام‌های چتی مخصوص Qwen2-VL را می‌سازد.
    فقط از 'ترکیب سوال و گزینه ها' استفاده می‌کنیم و از مدل می‌خواهیم یک عدد 1..4 برگرداند.
    """
    system_text = (
        "تو یک دستیار پاسخ‌گوی چندگزینه‌ای هستی. "
        "فقط و فقط با یک عدد از 1 تا 4 پاسخ بده؛ هیچ متن دیگری ننویس."
    )

    user_suffix = "\n\nلطفاً فقط با یک عدد از ۱ تا ۴ پاسخ بده."
    content = []
    images = []
    if img is not None:
        content.append({"type": "image", "image": img})
        images.append(img)
    content.append({"type": "text", "text": full_prompt_text + user_suffix})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": content},
    ]
    return messages, images

def run_row(processor, model, device, row_id: int, row: pd.Series) -> dict:
    # ستون‌های فارسی را پیدا کنیم
    # اجباری: «ترکیب سوال و گزینه ها»
    possible_combo_cols = [c for c in row.index if "ترکیب" in str(c)]
    if not possible_combo_cols:
        raise KeyError("ستون «ترکیب سوال و گزینه ها» در CSV پیدا نشد.")
    combo_col = possible_combo_cols[0]
    full_prompt_text = str(row[combo_col]).strip()

    # گزینه‌ها و پاسخ طلایی
    opt_cols = [c for c in row.index if "گزینه" in str(c)]
    # مرتب‌سازی گزینه‌ها بر اساس یک..چهار
    name_to_idx = {"یک":1, "دو":2, "سه":3, "چهار":4}
    def opt_key(cname):
        for k, v in name_to_idx.items():
            if k in str(cname):
                return v
        # fallback: سعی کن انتهای عدد انگلیسی را بیابی
        m = re.search(r"(\d+)$", str(cname))
        return int(m.group(1)) if m else 999
    opt_cols = sorted(opt_cols, key=opt_key)[:4]
    options = [str(row[c]).strip() for c in opt_cols]

    # پاسخ درست (شماره 1..4)
    gold_idx = None
    for key in row.index:
        if "پاسخ" in str(key):
            try:
                gold_idx = int(str(row[key]).strip().replace(" ", ""))
            except Exception:
                pass
            break

    # تصویر (اختیاری)
    img = None
    for key in row.index:
        if "عکس" in str(key):
            img = smart_open_image(row[key])
            break

    # ساخت پیام‌ها و اجرای مدل
    messages, images = build_messages(full_prompt_text, img)

    # Qwen2-VL chat template -> inputs
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=images if images else None,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    # فقط بخش تولیدی را بگیریم
    gen_ids = generated[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # استخراج پاسخ 1..4
    pred_idx = extract_digit_1_to_4(output_text)
    if pred_idx is None:
        pred_idx = pick_by_fuzzy_match(output_text, options)

    pred_label = options[pred_idx - 1] if pred_idx and 1 <= pred_idx <= 4 else ""

    return {
        "row_id": row_id,
        "question_used": full_prompt_text,
        "image_used": "yes" if img is not None else "no",
        "option1": options[0] if len(options) > 0 else "",
        "option2": options[1] if len(options) > 1 else "",
        "option3": options[2] if len(options) > 2 else "",
        "option4": options[3] if len(options) > 3 else "",
        "gold_index": gold_idx,
        "pred_index": pred_idx,
        "pred_label": pred_label,
        "is_correct": (pred_idx == gold_idx) if (pred_idx is not None and gold_idx is not None) else None,
        "model_output_raw": output_text,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF repo id, e.g., Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--input", type=str, default=str(INPUT_CSV), help="Path to input CSV")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV), help="Path to output CSV")
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    print(f"📥 Loading CSV: {input_csv}")
    # utf-8-sig برای BOM احتمالی
    df = pd.read_csv(input_csv, encoding="utf-8-sig", engine="python")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Loading model: {args.model} on {device}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    processor = AutoProcessor.from_pretrained(args.model)

    results = []
    for i, row in df.iterrows():
        try:
            res = run_row(processor, model, device, i, row)
        except Exception as e:
            res = {
                "row_id": i, "question_used": "", "image_used": "",
                "option1":"", "option2":"", "option3":"", "option4":"",
                "gold_index": None, "pred_index": None, "pred_label": "",
                "is_correct": None, "model_output_raw": f"ERROR: {e}"
            }
        results.append(res)
        if (i+1) % 5 == 0:
            print(f"… processed {i+1} rows")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved predictions to: {output_csv} | {len(out_df)} rows")

if __name__ == "__main__":
    main()
