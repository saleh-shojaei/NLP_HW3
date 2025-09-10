import os
import re
from PIL import Image, ImageOps, UnidentifiedImageError

input_root = "./images"
output_root = "./processed"

allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
target_size = (224, 224)

os.makedirs(output_root, exist_ok=True)

def clean_filename(filename):
    cleaned = re.sub(r'\s+', '_', filename)
    
    cleaned = re.sub(r'[<>:"/\\|?*]', '', cleaned)

    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
    
    cleaned = re.sub(r'_+', '_', cleaned)
    
    cleaned = cleaned.strip('_')
    
    return cleaned

def center_crop_to_square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2))

def ensure_rgb(img):
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img

saved_names = set()

for root, _, files in os.walk(input_root):
    rel = os.path.relpath(root, input_root)
    out_dir = os.path.join(output_root, rel) if rel != "." else output_root
    os.makedirs(out_dir, exist_ok=True)

    for fname in files:
        src = os.path.join(root, fname)
        base, ext = os.path.splitext(fname)
        if ext.lower() not in allowed_ext:
            print(f"⏭️ Skipped (non-image): {fname}")
            continue

        cleaned_base = clean_filename(base)
        if not cleaned_base:
            cleaned_base = "unnamed"
            print(f"⚠️ Empty filename after cleaning, using 'unnamed': {fname}")

        if cleaned_base in saved_names:
            print(f"⏭️ Skipped duplicate name: {cleaned_base}")
            continue

        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                im = center_crop_to_square(im)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                im = im.resize(target_size, resample=resample)
                im = ensure_rgb(im)

                out_path = os.path.join(out_dir, f"{cleaned_base}.jpg")
                im.save(out_path, "JPEG", quality=90, optimize=True, subsampling="medium")

                saved_names.add(cleaned_base)
                print(f"✅ Processed: {src} -> {out_path}")

        except (UnidentifiedImageError, OSError) as e:
            print(f"⚠️ Error processing {fname}: {e}")