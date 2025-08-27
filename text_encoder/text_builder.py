# text_builder.py
from typing import Any, Dict, List
import re

def _norm_space(s: str) -> str:
    """تمیز کردن فاصله‌ها و نویزهای ساده."""
    s = s.replace("\u200c", "\u200c")  # اجازه می‌ده نیم‌فاصله فارسی حفظ بشه
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s

def _fmt_amount(amount: Any, unit: str) -> str:
    """نمایش مقدار + واحد به‌صورت جمع‌وجور. اگر خالی بود، رشته‌ی خالی بده."""
    if amount is None or amount == "":
        return ""
    # اگر عدد اعشاری با .0 بود، تمیزش کن
    if isinstance(amount, float) and amount.is_integer():
        amount = int(amount)
    if unit:
        return f"{amount} {unit}".strip()
    return f"{amount}".strip()

def _join_list_inline(items: List[str], sep: str = "، ") -> str:
    items = [i for i in (item.strip() for item in items) if i]
    return sep.join(items)

def build_text_from_item(item: Dict[str, Any]) -> str:
    """
    از ساختار JSON مورد نظر، یک رشته‌ی فارسی یک‌تکه می‌سازد.
    اگر کلیدی نبود، همان بخش در خروجی خالی می‌ماند.
    فیلدها: title, province (از location.province), ingredients[], instructions[]
    """
    title = _norm_space(str(item.get("title", "") or ""))
    # لوکیشن/استان
    province = ""
    loc = item.get("location") or {}
    if isinstance(loc, dict):
        province = _norm_space(str((loc.get("province") or "")))

    # مواد لازم
    ingredients = item.get("ingredients") or []
    ing_lines: List[str] = []
    if isinstance(ingredients, list):
        for ing in ingredients:
            if not isinstance(ing, dict):
                continue
            name = _norm_space(str(ing.get("name", "") or ""))
            amount = ing.get("amount", "")
            unit = _norm_space(str(ing.get("unit", "") or ""))
            amt = _fmt_amount(amount, unit)
            if name and amt:
                ing_lines.append(f"{name} ({amt})")
            elif name:
                ing_lines.append(name)
    ing_str = _join_list_inline(ing_lines)

    # دستور پخت
    instructions = item.get("instructions") or []
    inst_lines: List[str] = []
    if isinstance(instructions, list):
        # همه را یک خطی و تمیز کن
        for idx, step in enumerate(instructions, start=1):
            if not isinstance(step, str):
                continue
            st = _norm_space(step)
            # اگر شماره‌گذاری اولش هست (مثل "۱." یا "1." یا "۱)"…), حذف کن تا خودمان شماره‌گذاری کنیم
            st = re.sub(r"^\s*[\d۱۲۳۴۵۶۷۸۹0]+\s*[\.\)\-ـ]\s*", "", st)
            inst_lines.append(f"{idx}) {st}")
    inst_str = _join_list_inline(inst_lines, sep="؛ ")

    # مونتاژ نهایی (یک رشته‌ی یک‌تکه)
    sections = []
    # عنوان + استان را در یک خط می‌آوریم
    header_parts = []
    if title:
        header_parts.append(f"عنوان: {title}")
    if province:
        header_parts.append(f"استان: {province}")
    if header_parts:
        sections.append(" | ".join(header_parts))

    if ing_str:
        sections.append(f"مواد لازم: {ing_str}")
    if inst_str:
        sections.append(f"دستور پخت: {inst_str}")

    final_text = " \n".join(sections).strip()
    return final_text
