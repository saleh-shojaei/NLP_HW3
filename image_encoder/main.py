# main.py
from pathlib import Path
from emb import encode_image
from write_db import add_embedding

IMAGES_DIR = "../processed_images"   # مسیر پوشه‌ی تصاویر .jpg

def make_db_id_from_filename(image_path: str) -> str:
    """
    فرمت نام فایل:
      <dish_id>_<seq>_<title>__<label>.jpg
    خروجی ID: "<dish_id>_<seq>"
    """
    stem = Path(image_path).stem                  # "6_3_قیقاناخ__2 مرحله"
    left = stem.split("__", 1)[0]                 # "6_3_قیقاناخ"
    parts = left.split("_")
    dish_id = parts[0] if parts else "unknown"
    seq = parts[1] if len(parts) > 1 and parts[1].isdigit() else "1"
    return f"{dish_id}_{seq}"

def main():
    img_paths = sorted(Path(IMAGES_DIR).glob("*.jpg"))
    if not img_paths:
        print(f"No .jpg files found in: {IMAGES_DIR}")
        return

    total, ok, fail = len(img_paths), 0, 0
    print(f"Found {total} images in '{IMAGES_DIR}'. Encoding & writing to DB...")

    for p in img_paths:
        try:
            db_id = make_db_id_from_filename(str(p))
            emb = encode_image(str(p))  # 512-d float32 unit vector
            add_embedding(db_id, emb)   # فقط id + embedding
            ok += 1
            print(f"[OK] id='{db_id}'  file='{p.name}'")
        except Exception as e:
            fail += 1
            print(f"[FAIL] file='{p.name}' -> {e}")

    print(f"\nDone. Total: {total}, OK: {ok}, Fail: {fail}")

if __name__ == "__main__":
    main()
