# main_text_from_json.py
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from text_builder import build_text_from_item
from emb_text import encode_text
from write_db_text import add_embedding  # کالکشن جدا برای متن

JSON_PATH = "../Foods-dataset/foods.normalized.json"


def load_items(json_path: str) -> List[Dict[str, Any]]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # اگر به‌صورت دیکشنری باشد و داخلش لیست داشته باشد
        for v in data.values():
            if isinstance(v, list):
                return v
        # در غیر این صورت یک آیتم تکی
        return [data]
    raise ValueError("Unsupported JSON structure: expected list or dict.")


def make_db_id_from_item(item: Dict[str, Any]) -> str:
    """
    با توجه به اینکه همه آیتم‌ها حتماً فیلد id دارند.
    """
    return f"{item['id']}_txt"


def should_skip_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    اگر ingredients یا instructions دقیقا لیست خالی باشند، True و دلیلش را برگردان.
    فقط دقیقا [] را در نظر می‌گیریم (مطابق درخواست).
    """
    if "ingredients" in item and isinstance(item["ingredients"], list) and len(item["ingredients"]) == 0:
        return True, "empty ingredients []"
    if "instructions" in item and isinstance(item["instructions"], list) and len(item["instructions"]) == 0:
        return True, "empty instructions []"
    return False, ""


def main():
    items = load_items(JSON_PATH)
    total, ok, skip, fail = len(items), 0, 0, 0
    print(f"Found {total} items in '{JSON_PATH}'. Building text, encoding, and writing to TEXT collection...")

    for it in items:
        try:
            # تضمین وجود id طبق فرض مسئله
            if "id" not in it:
                raise KeyError("Item has no 'id' field.")

            # ـــ فیلتر موردنظر: اگر ingredients یا instructions دقیقا [] باشند، اسکپ کن
            do_skip, reason = should_skip_item(it)
            if do_skip:
                skip += 1
                print(f"[SKIP] id='{it.get('id')}'  title='{it.get('title','')}'  reason={reason}")
                continue

            db_id = make_db_id_from_item(it)
            text = build_text_from_item(it)
            if not text.strip():
                raise ValueError("Built text is empty.")

            # پیش‌نمایش کوتاه برای لاگ
            preview = text if len(text) <= 80 else text[:77] + "..."
            print(f"[BUILD] id='{db_id}'  title='{it.get('title','')}'  text='{preview}'")

            emb = encode_text(text)  # np.ndarray (d,) L2-normalized
            meta = {
                "type": "text",
                "dish_id": it["id"],
                "title": it.get("title", ""),
                "province": (it.get("location", {}) or {}).get("province", ""),
            }
            add_embedding(db_id, emb, metadata=meta)
            ok += 1
            print(f"[OK]   id='{db_id}'  title='{meta['title']}'  embedded")

        except Exception as e:
            fail += 1
            print(f"[FAIL] id='{it.get('id', '?')}' -> {e}")

    print(f"\nDone. Total: {total}, OK: {ok}, SKIP: {skip}, Fail: {fail}")


if __name__ == "__main__":
    main()
