"""Split YOLO dataset from raw/train into YOLOv8 train/val folders."""

import random
import shutil
from pathlib import Path

random.seed(42)

raw_dir = Path("raw/train")
images_train = Path("dataset/images/train")
images_val = Path("dataset/images/val")
labels_train = Path("dataset/labels/train")
labels_val = Path("dataset/labels/val")

extensions = (".png", ".jpg", ".jpeg")
pairs = []
missing_count = 0

for img_path in raw_dir.iterdir():
    if img_path.suffix.lower() not in extensions:
        continue
    stem = img_path.stem
    txt_path = raw_dir / f"{stem}.txt"
    if txt_path.exists():
        pairs.append((img_path, txt_path))
    else:
        missing_count += 1

random.shuffle(pairs)
n = len(pairs)
split_idx = int(n * 0.8)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

for img_path, txt_path in train_pairs:
    shutil.copy2(img_path, images_train / img_path.name)
    shutil.copy2(txt_path, labels_train / txt_path.name)

for img_path, txt_path in val_pairs:
    shutil.copy2(img_path, images_val / img_path.name)
    shutil.copy2(txt_path, labels_val / txt_path.name)

print("Summary:")
print(f"  Total pairs:     {n}")
print(f"  Train pairs:     {len(train_pairs)}")
print(f"  Val pairs:       {len(val_pairs)}")
print(f"  Missing labels:  {missing_count}")
