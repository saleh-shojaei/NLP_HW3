import os
import json
import re
import time
import pathlib
import mimetypes
from urllib.parse import urlparse, unquote
from typing import Any, Dict, List, Tuple
from collections import defaultdict

try:
    import requests
    from requests.exceptions import RequestException
except ImportError:
    raise SystemExit("Please install requests first:  pip install requests")

# ===== تنظیم مسیرها =====
AGGREGATED_PATH = "../Foods-dataset/aggregated.json"  # فقط از این فایل می‌خوانیم
IMAGES_DIR      = "images"           # پوشه‌ی خروجی برای عکس‌ها
MANIFEST_NAME   = "manifest.json"    # نگاشت ساده: filename -> title
FETCH_LOG       = "fetch-images.logs"
FAIL_LOG        = "failures.logs"

# ===== ابزارها =====
def safe_name(text: Any, max_len: int = 120) -> str:
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
        "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
        "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp",
        "image/tiff": ".tiff", "image/svg+xml": ".svg",
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

# ===== Cookpad rewrite =====
_COOKPAD_STEP_RE = re.compile(r"/step_attachment/images/([A-Za-z0-9]+)", re.IGNORECASE)

def _rewrite_cookpad(url: str) -> str:
    try:
        parsed = urlparse(url)
        if "cookpad.com" in parsed.netloc.lower():
            m = _COOKPAD_STEP_RE.search(parsed.path)
            if m:
                step_id = m.group(1)
                return f"https://img-global.cpcdn.com/steps/{step_id}/640x640sq80/photo.webp"
    except Exception:
        pass
    return url

# ===== دانلود با ریتـرای =====
def download(url: str, timeout: int = 30, retries: int = 3, backoff: float = 1.0) -> Tuple[bytes, str, str]:
    """
    دانلود با ریتـرای. خروجی: (content_bytes, mime, final_url)
    """
    final_url = _rewrite_cookpad(url)
    headers = {"User-Agent": "Mozilla/5.0 (NLP_HW3 Image Fetcher)"}
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(final_url, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                mime = r.headers.get("Content-Type", "").split(";")[0].strip()
                return r.content, mime, final_url
        except RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * attempt)  # backoff افزایشی ساده
            else:
                raise e
    # نباید به اینجا برسیم
    raise last_exc or RuntimeError("unknown download error")

# ===== لاگ =====
class DualLogger:
    def __init__(self, fetch_log_path: str, fail_log_path: str):
        self.fetch_fh = open(fetch_log_path, "w", encoding="utf-8")
        self.fail_fh  = open(fail_log_path,  "w", encoding="utf-8")
    def log(self, msg: str = ""):
        print(msg)
        self.fetch_fh.write(msg + ("\n" if not msg.endswith("\n") else ""))
        self.fetch_fh.flush()
    def record_failure(self, title: str, url: str):
        self.fail_fh.write(f"{title}\t{url}\n")
        self.fail_fh.flush()
    def close(self):
        try:
            self.fetch_fh.close()
        finally:
            self.fail_fh.close()

def main():
    logger = DualLogger(FETCH_LOG, FAIL_LOG)
    try:
        out_dir = pathlib.Path(IMAGES_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / MANIFEST_NAME
        file_to_title: Dict[str, str] = {}

        # فقط aggregated.json
        if not os.path.exists(AGGREGATED_PATH):
            logger.log(f"! aggregated file not found: {AGGREGATED_PATH}")
            return

        with open(AGGREGATED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            logger.log("! aggregated.json must be a list or an object")
            return

        total_found = 0
        total_saved = 0
        errors = 0

        logger.log(f"=== Source: {AGGREGATED_PATH} (records: {len(data)}) ===")

        # شمارنده ترتیبی per-id
        id_seq_counter = defaultdict(int)

        for idx, rec in enumerate(data, start=1):
            if not isinstance(rec, dict):
                continue

            rid = rec.get("id", f"record_{idx}")
            rid_stem = safe_name(rid)
            title = safe_name(rec.get("title", f"record_{idx}"))

            images_val = extract_images_value(rec)
            pairs = iter_image_entries(images_val)
            if not pairs:
                continue

            logger.log(f"\n- Title: {title} (id={rid})")
            logger.log("  images:")
            for label, url in pairs:
                logger.log(f"    • {label}: {url}")
            total_found += len(pairs)

            for label, url in pairs:
                # افزایش شمارنده‌ی این id (چه تصویر دانلود شود چه placeholder بسازیم)
                id_seq_counter[rid_stem] += 1
                seq = id_seq_counter[rid_stem]

                label_stem = safe_name(label)

                # --- NEW: هندل URLهای خالی/بی‌اعتبار ---
                if not isinstance(url, str) or not url.strip() or not url.lower().startswith("http"):
                    placeholder = out_dir / f"{rid_stem}_{seq}_{title}__{label_stem}.missing.txt"
                    try:
                        with open(placeholder, "w", encoding="utf-8") as ph:
                            ph.write(f"Missing/empty URL for title='{title}', label='{label}'\n")
                        file_to_title[placeholder.name] = title
                        logger.log(f"    · skipped empty URL -> created placeholder: {placeholder.name}")
                    except Exception as e:
                        logger.log(f"    ! placeholder write failed: {placeholder} -> {e}")
                        logger.record_failure(title, str(url))
                        errors += 1
                    continue
                # --------------------------------------

                try:
                    content, mime, final_url = download(url)
                except Exception as e:
                    logger.log(f"    ! download failed: {url} -> {e}")
                    logger.record_failure(title, url)
                    errors += 1
                    continue

                ext = guess_ext_from_url(final_url) or guess_ext_from_mime(mime)
                if not ext:
                    sig = content[:12]
                    if sig.startswith(b"\xff\xd8"): ext = ".jpg"
                    elif sig.startswith(b"\x89PNG\r\n\x1a\n"): ext = ".png"
                    elif sig[:3] == b"GIF": ext = ".gif"
                    elif sig[:4] == b"RIFF" and b"WEBP" in sig: ext = ".webp"
                    else: ext = ".bin"

                filename = f"{rid_stem}_{seq}_{title}__{label_stem}{ext}"
                out_path = ensure_unique(out_dir / filename)

                try:
                    with open(out_path, "wb") as fp:
                        fp.write(content)
                    file_to_title[out_path.name] = title
                    total_saved += 1
                    logger.log(f"    + saved: {out_path.name}")
                except Exception as e:
                    logger.log(f"    ! save failed: {out_path} -> {e}")
                    logger.record_failure(title, url)
                    errors += 1

        # مانیفست ساده
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
