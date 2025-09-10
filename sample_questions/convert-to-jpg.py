from pathlib import Path
from PIL import Image

# مسیر پوشه عکس‌ها
folder = Path(r"./")   # پوشه‌ات رو اینجا بذار

for img_path in folder.glob("*.jpeg"):
    try:
        img = Image.open(img_path)
        new_path = img_path.with_suffix(".jpg")
        img.convert("RGB").save(new_path, "JPEG")
        img.close()
        img_path.unlink()  # حذف فایل اصلی .jpeg
        print(f"Converted: {img_path.name} -> {new_path.name}")
    except Exception as e:
        print(f"Error with {img_path}: {e}")
