import os
import json
import re
import pathlib
import mimetypes
from urllib.parse import urlparse, unquote
from typing import Any, Dict, List, Tuple

try:
    import requests
except ImportError:
    raise SystemExit("Please install requests first:  pip install requests")

# مسیرها
DATASET_DIR   = "Foods-dataset"           # پوشه‌ی ورودی
IMAGES_DIR    = "images"            # پوشه‌ی خروجی برای عکس‌ها
MANIFEST_NAME = "manifest.json"     # نام فایل نگاشت
FETCH_LOG     = "fetch-images.logs" # لاگ کامل خروجی
FAIL_LOG      = "failures.logs"     # فقط موارد ناموفق (Title + URL)

# ————— ابزارها —————
def safe_name(text: str, max_len: int = 120) -> str:
    """نام فایل سازگار با فایل‌سیستم (حروف فارسی حفظ می‌شوند)."""
    text = str(text)
    text = text.replace("/", " ").replace("\\", " ")
    text = re.sub(r'[\x00-\x1f<>:"|?*]+', " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:max_len].rstrip() or "untitled")

def guess_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    name = pathlib.Path(unquote(path)).name
    ext = pathlib.Path(name).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}:
        return ext
    return ""

def guess_ext_from_mime(mime: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/svg+xml": ".svg",
    }
    if mime in mapping:
        return mapping[mime]
    return mimetypes.guess_extension(mime or "") or ""

def ensure_unique(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        return path
    base, ext = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{base}-{i}{ext}")
        if not candidate.exists():
            return candidate
        i += 1

def extract_images_value(record: Dict[str, Any]) -> Any:
    """
    فیلد images را برمی‌گرداند.
    اگر رشته‌ی JSON باشد، به dict/list تبدیلش می‌کند؛
    در غیر اینصورت همان مقدار را برمی‌گرداند (ممکن است dict/list یا رشته‌ی ساده باشد).
    """
    if "images" not in record:
        return None
    val = record["images"]
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val

def iter_image_entries(images_field: Any) -> List[Tuple[str, str]]:
    """
    از فیلد images، زوج‌های (label, url) استخراج می‌کند.
    پشتیبانی از dict و list و رشته‌ی تکی.
    """
    pairs: List[Tuple[str, str]] = []
    if images_field is None:
        return pairs

    if isinstance(images_field, dict):
        for k, v in images_field.items():
            if isinstance(v, str):
                pairs.append((str(k), v))
            elif isinstance(v, list):
                for i, url in enumerate(v, start=1):
                    if isinstance(url, str):
                        pairs.append((f"{k}_{i}", url))
    elif isinstance(images_field, list):
        for i, v in enumerate(images_field, start=1):
            if isinstance(v, str):
                pairs.append((f"img_{i}", v))
    elif isinstance(images_field, str):
        pairs.append(("image", images_field))
    return pairs

def download(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    headers = {"User-Agent": "Mozilla/5.0 (NLP_HW3 Image Fetcher)"}
    with requests.get(url, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        mime = r.headers.get("Content-Type", "").split(";")[0].strip()
        return r.content, mime

# ——— logging helpers ———
class DualLogger:
    def __init__(self, fetch_log_path: str, fail_log_path: str):
        self.fetch_fh = open(fetch_log_path, "w", encoding="utf-8")
        self.fail_fh  = open(fail_log_path,  "w", encoding="utf-8")

    def log(self, msg: str = ""):
        print(msg)
        self.fetch_fh.write(msg + ("\n" if not msg.endswith("\n") else ""))
        self.fetch_fh.flush()

    def record_failure(self, title: str, url: str):
        # فقط عنوان و لینک (طبق درخواست)
        line = f"{title}\t{url}"
        self.fail_fh.write(line + "\n")
        self.fail_fh.flush()

    def close(self):
        try:
            self.fetch_fh.close()
        finally:
            self.fail_fh.close()

def main():
    logger = DualLogger(FETCH_LOG, FAIL_LOG)
    try:
        # آماده‌سازی خروجی
        out_dir = pathlib.Path(IMAGES_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / MANIFEST_NAME
        file_to_title: Dict[str, str] = {}

        # پیمایش همه فایل‌های json در dataset (به‌همراه زیردایرکتوری‌ها)
        json_files: List[str] = []
        for root, _, files in os.walk(DATASET_DIR):
            for name in files:
                if name.lower().endswith(".json"):
                    json_files.append(os.path.join(root, name))

        if not json_files:
            logger.log(f"No JSON files found under '{DATASET_DIR}'")
            return

        total_found = 0
        total_saved = 0
        errors = 0

        for jf in json_files:
            logger.log(f"\n=== File: {jf} ===")
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.log(f"  ! JSON read error: {e}")
                errors += 1
                continue

            if not isinstance(data, list):
                logger.log("  ! Expected a list of records; skipping this file.")
                continue

            for idx, rec in enumerate(data, start=1):
                if not isinstance(rec, dict):
                    continue
                title = safe_name(rec.get("title", f"record_{idx}"))
                images_val = extract_images_value(rec)
                pairs = iter_image_entries(images_val)

                if not pairs:
                    continue

                logger.log(f"\n  - Title: {title}")
                logger.log("    images:")
                for label, url in pairs:
                    logger.log(f"      • {label}: {url}")
                total_found += len(pairs)

                # دانلود هر URL
                for i, (label, url) in enumerate(pairs, start=1):
                    try:
                        content, mime = download(url)
                    except Exception as e:
                        logger.log(f"      ! download failed: {url} -> {e}")
                        logger.record_failure(title, url)
                        errors += 1
                        continue

                    # تعیین نام فایل: title + label
                    label_stem = safe_name(label)
                    ext = guess_ext_from_url(url) or guess_ext_from_mime(mime)
                    if not ext:
                        sig = content[:12]
                        if sig.startswith(b"\xff\xd8"):
                            ext = ".jpg"
                        elif sig.startswith(b"\x89PNG\r\n\x1a\n"):
                            ext = ".png"
                        elif sig[:3] == b"GIF":
                            ext = ".gif"
                        elif sig[:4] == b"RIFF" and b"WEBP" in sig:
                            ext = ".webp"
                        else:
                            ext = ".bin"

                    filename = f"{title}__{label_stem}{ext}"
                    out_path = ensure_unique(out_dir / filename)

                    try:
                        with open(out_path, "wb") as fp:
                            fp.write(content)
                        file_to_title[out_path.name] = title
                        total_saved += 1
                        logger.log(f"      + saved: {out_path.name}")
                    except Exception as e:
                        logger.log(f"      ! save failed: {out_path} -> {e}")
                        # دانلود موفق بوده، ولی ذخیره ناموفق؛ همچنان به‌عنوان failure منطقی محسوب می‌کنیم
                        logger.record_failure(title, url)
                        errors += 1

        # نوشتن مانیفست
        try:
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(file_to_title, mf, ensure_ascii=False, indent=2)
            logger.log(f"\nManifest written: {manifest_path}  (items: {len(file_to_title)})")
        except Exception as e:
            logger.log(f"\n! manifest write failed: {e}")
            errors += 1

        logger.log(f"\nDone. Found URLs: {total_found}, Saved: {total_saved}, Errors: {errors}")
        logger.log(f"Logs: {FETCH_LOG}  |  Failures: {FAIL_LOG}")
    finally:
        logger.close()

if __name__ == "__main__":
    main()
