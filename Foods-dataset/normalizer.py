# -*- coding: utf-8 -*-
"""
normalize_fa_json.py
- همهٔ رشته‌ها (keys و values) را در یک ساختار JSON به‌صورت بازگشتی نرمال می‌کند.
- تبدیل \n به فاصله در تمام رشته‌ها (به جز images)، حروف عربی -> فارسی، حذف اعراب/کشیده،
  مرتب‌سازی فاصله‌ها و نیم‌فاصله‌ها، حذف ایموجی‌ها،
  و استفاده از hazm.Normalizer برای ریزه‌کاری‌های نگارشی.
"""

import json
import re
import unicodedata
from typing import Any
import regex as reg  # بهتر از re برای یونیکد
from hazm.normalizer import Normalizer  # ایمپورت مستقیم تا gensim لود نشود

# ---------- تنظیمات ----------

PERSIANIZE_DIGITS = True
NORMALIZE_KEYS    = True

# ---------- جداول و الگوها ----------

AR2FA_CHAR_MAP = str.maketrans({
    "\u064a": "ی",
    "\u0649": "ی",
    "\u06cc": "ی",
    "\u0643": "ک",
    "\u06a9": "ک",
    "\u0629": "ه",
    "\u06c0": "ۀ",
    "\u0626": "ئ",
    "\u0643": "ک",
    "\u0670": "",
})

ZWJ      = "\u200d"
ZWNJ     = "\u200c"
LRE      = "\u202A"
RLE      = "\u202B"
PDF      = "\u202C"
LRM      = "\u200E"
RLM      = "\u200F"
TATWEEL  = "\u0640"

NOISE_CHARS = "".join([LRE, RLE, PDF, LRM, RLM, TATWEEL, "\u2066", "\u2067", "\u2069"])

RE_SPACES         = reg.compile(r"[ \t\u00A0\u2007\u202F]+")
RE_MULTI_ZWNJ     = reg.compile(rf"{ZWNJ}{{2,}}")
RE_ZW_AROUND_PUNC = reg.compile(rf"{ZWNJ}*([،؛،,:;.!؟\)]){ZWNJ}*")
RE_AROUND_AFFIX   = reg.compile(rf"\s*{ZWNJ}\s*")

RE_EMOJI = reg.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF"
    r"\U0001F1E6-\U0001F1FF\U00002500-\U00002BEF]"
)

def remove_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if unicodedata.category(ch) != "Mn")

RE_PERSIAN_LETTER = reg.compile(r"[\p{Arabic}&&[^\u060C]]")

def looks_persian(s: str) -> bool:
    return bool(RE_PERSIAN_LETTER.search(s))

hazm_norm = Normalizer(
    persian_numbers=PERSIANIZE_DIGITS,
    remove_diacritics=True,
)

# ---------- توابع پاک‌سازی ----------

def basic_char_normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = RE_EMOJI.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(AR2FA_CHAR_MAP)
    s = remove_diacritics(s)
    s = s.translate(str.maketrans("", "", NOISE_CHARS))
    s = RE_SPACES.sub(" ", s)
    s = RE_MULTI_ZWNJ.sub(ZWNJ, s)
    s = RE_ZW_AROUND_PUNC.sub(r"\1", s)
    s = re.sub(r"\s*([،؛,:;.!؟])\s*", r"\1 ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"[ \u2009]{2,}", " ", s).strip()
    return s

def finalize_with_hazm(s: str) -> str:
    s = hazm_norm.normalize(s)
    if looks_persian(s):
        s = s.replace(",", "،")
        s = re.sub(r"\s*،\s*", "، ", s)
    s = RE_AROUND_AFFIX.sub(ZWNJ, s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def normalize_text(s: str) -> str:
    s = basic_char_normalize(s)
    s = finalize_with_hazm(s)
    return s

# ---------- پیمایش بازگشتی JSON ----------

def normalize_json(obj: Any, normalize_keys: bool = NORMALIZE_KEYS, parent_key: str = "") -> Any:
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # اگر کلید images است، بدون تغییر بریز داخل
            if isinstance(k, str) and k == "images":
                new_dict[k] = v
                continue
            new_k = normalize_text(k) if (normalize_keys and isinstance(k, str)) else k
            new_dict[new_k] = normalize_json(v, normalize_keys, parent_key=new_k)
        return new_dict

    elif isinstance(obj, list):
        return [normalize_json(x, normalize_keys, parent_key=parent_key) for x in obj]

    elif isinstance(obj, tuple):
        return tuple(normalize_json(x, normalize_keys, parent_key=parent_key) for x in obj)

    elif isinstance(obj, str):
        # اگر والد images بود → رشته دست‌نخورده برگرده
        if parent_key == "images":
            return obj
        return normalize_text(obj)

    else:
        return obj

# ---------- مثال استفاده ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Normalize Persian texts inside a JSON file.")
    ap.add_argument("input", help="مسیر فایل ورودی JSON")
    ap.add_argument("-o", "--output", help="مسیر فایل خروجی JSON (پیش‌فرض: افزودن .normalized)")
    ap.add_argument("--keep-keys", action="store_true", help="کلیدهای دیکشنری را نرمال نکن")
    ap.add_argument("--latin-digits", action="store_true", help="اعداد را فارسی نکن (لاتین نگه دار)")
    args = ap.parse_args()

    if args.keep_keys:
        NORMALIZE_KEYS = False
    if args.latin_digits:
        PERSIANIZE_DIGITS = False
        hazm_norm = Normalizer(persian_numbers=False, remove_diacritics=True)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = normalize_json(data, normalize_keys=NORMALIZE_KEYS)

    out_path = args.output or (args.input.rsplit(".", 1)[0] + ".normalized.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"✅ فایل نرمال‌شده ذخیره شد: {out_path}")
