# -*- coding: utf-8 -*-
"""
normalize_fa_json.py
- همهٔ رشته‌ها (keys و values) را در یک ساختار JSON به‌صورت بازگشتی نرمال می‌کند.
- حروف عربی -> فارسی، حذف اعراب/کشیده، مرتب‌سازی فاصله‌ها و نیم‌فاصله‌ها،
  و استفاده از hazm.Normalizer برای ریزه‌کاری‌های نگارشی.
"""

import json
import re
import unicodedata
from copy import deepcopy
from typing import Any
import regex as reg  # بهتر از re برای یونیکد
from hazm import Normalizer

# ---------- تنظیمات ----------

PERSIANIZE_DIGITS = True     # اگر False کنید، اعداد دست‌نخورده می‌مانند
NORMALIZE_KEYS     = True     # اگر نمی‌خواهید کلیدهای دیکشنری تغییر کنند، False کنید

# ---------- جداول و الگوها ----------

# نگاشت حروف عربی به فارسی
AR2FA_CHAR_MAP = str.maketrans({
    "\u064a": "ی",  # ي -> ی
    "\u0649": "ی",  # ى -> ی
    "\u06cc": "ی",  # ي/ى به هر حال -> ی (تثبیت)
    "\u0643": "ک",  # ك -> ک
    "\u06a9": "ک",  # ک
    "\u0629": "ه",  # ة -> ه (در اغلب متون عمومی)
    "\u06c0": "ۀ",  # ۀ
    "\u0626": "ئ",  # ی همزه‌دار
    "\u0643": "ک",
    "\u0670": "",   # superscript alef (اعراب)
})

# کاراکترهای زائد/نویز
ZWJ      = "\u200d"  # Zero Width Joiner
ZWNJ     = "\u200c"  # Zero Width Non-Joiner
LRE      = "\u202A"
RLE      = "\u202B"
PDF      = "\u202C"
LRM      = "\u200E"
RLM      = "\u200F"
TATWEEL  = "\u0640"  # کشیده

NOISE_CHARS = "".join([LRE, RLE, PDF, LRM, RLM, TATWEEL, "\u2066", "\u2067", "\u2069"])

# الگوهای فاصله/نیم‌فاصله
RE_SPACES = reg.compile(r"[ \t\u00A0\u2007\u202F]+")  # انواع space ها
RE_MULTI_ZWNJ = reg.compile(rf"{ZWNJ}{{2,}}")
RE_ZW_AROUND_PUNC = reg.compile(rf"{ZWNJ}*([،؛،,:;.!؟\)]){ZWNJ}*")

# الگوی ترکیب‌های فارسی/انگلیسی برای نیم‌فاصله‌های اشتباه
RE_AROUND_AFFIX = reg.compile(rf"\s*{ZWNJ}\s*")

# حذف نشانه‌های اعراب (combining marks)
def remove_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if unicodedata.category(ch) != "Mn")

# تشخیص «احتمال فارسی بودن» (برای اعداد/ویرگول و…)
RE_PERSIAN_LETTER = reg.compile(r"[\p{Arabic}&&[^\u060C]]")  # هر چیز عربی/فارسی به جز ویرگول عربی

def looks_persian(s: str) -> bool:
    return bool(RE_PERSIAN_LETTER.search(s))

# نرمال‌ساز hazm
hazm_norm = Normalizer(
    persian_numbers=PERSIANIZE_DIGITS,
    remove_diacritics=True,
    # preserve_punctuation=True  # گزینه‌های دیگر به نیاز شما
)

# ---------- توابع پاک‌سازی سطح کاراکتر ----------

def basic_char_normalize(s: str) -> str:
    # یکسان‌سازی نمای یونیکد
    s = unicodedata.normalize("NFKC", s)
    # نگاشت عربی -> فارسی
    s = s.translate(AR2FA_CHAR_MAP)
    # حذف اعراب/کشیده/کنترل‌های جهت‌دهی
    s = remove_diacritics(s)
    s = s.translate(str.maketrans("", "", NOISE_CHARS))
    # فشرده‌سازی فاصله‌ها (فاصله معمولی)
    s = RE_SPACES.sub(" ", s)
    # فشرده‌سازی چند نیم‌فاصله
    s = RE_MULTI_ZWNJ.sub(ZWNJ, s)
    # نیم‌فاصله‌های اطراف علائم نگارشی حذف شود
    s = RE_ZW_AROUND_PUNC.sub(r"\1", s)
    # فاصله‌های اضافه اطراف علائم: قبل بدون فاصله، بعد یک فاصله (برای برخی نشانه‌ها)
    s = re.sub(r"\s*([،؛,:;.!؟])\s*", r"\1 ", s)
    # پرانتز باز: بعدش فاصله نیاید
    s = re.sub(r"\(\s+", "(", s)
    # قبل از پرانتز بسته فاصله حذف
    s = re.sub(r"\s+\)", ")", s)
    # فشرده‌سازی نهایی فاصله‌ها و trim
    s = re.sub(r"[ \u2009]{2,}", " ", s).strip()
    return s

def finalize_with_hazm(s: str) -> str:
    # فقط روی متونی که بوی فارسی می‌دهند، قاعده‌های ویژه را اعمال کن
    s = hazm_norm.normalize(s)
    # جایگزینی ویرگول انگلیسی با ویرگول فارسی، اگر متن فارسی است
    s = s.replace(",", "،")
    # اصلاح فواصل قبل/بعد ویرگول فارسی
    s = re.sub(r"\s*،\s*", "، ", s)
    # نیم‌فاصله‌های با فاصله اطرافشان
    s = RE_AROUND_AFFIX.sub(ZWNJ, s)
    # فشرده‌سازی نهایی
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def normalize_text(s: str) -> str:
    s = basic_char_normalize(s)
    s = finalize_with_hazm(s)
    return s

# ---------- پیمایش بازگشتی JSON ----------

def normalize_json(obj: Any, normalize_keys: bool = NORMALIZE_KEYS) -> Any:
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_k = normalize_text(k) if (normalize_keys and isinstance(k, str)) else k
            new_dict[new_k] = normalize_json(v, normalize_keys)
        return new_dict
    elif isinstance(obj, list):
        return [normalize_json(x, normalize_keys) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(normalize_json(x, normalize_keys) for x in obj)
    elif isinstance(obj, str):
        return normalize_text(obj)
    else:
        # سایر انواع (int/float/bool/None) دست‌نخورده
        return obj

# ---------- مثال استفاده: خواندن/نوشتن فایل ----------

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
        hazm_norm = Normalizer()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = normalize_json(data, normalize_keys=NORMALIZE_KEYS)

    out_path = args.output or (args.input.rsplit(".", 1)[0] + ".normalized.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"✅ فایل نرمال‌شده ذخیره شد: {out_path}")
