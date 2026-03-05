# camel-yolo-near-road: YOLOv8 camel detection + road segmentation (https://github.com/SugarV6/camel-yolo-near-road)
"""Ingest camel (detection) and road (segmentation) datasets from raw/ into final folder structure."""

import random
import shutil
from pathlib import Path

random.seed(42)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _find_label_for_image(img_path: Path, root: Path):
    """Return path to .txt label for image, or None. Tries same dir and images->labels sibling."""
    stem = img_path.stem
    # Same directory
    same_dir = img_path.parent / f"{stem}.txt"
    if same_dir.exists():
        return same_dir
    # Sibling labels/ (e.g. images/train/0.png -> labels/train/0.txt)
    parts = img_path.relative_to(root).parts
    if "images" in parts:
        i = parts.index("images")
        label_parts = parts[:i] + ("labels",) + parts[i + 1:]
        label_path = root.joinpath(*label_parts).with_suffix(".txt")
        if label_path.exists():
            return label_path
    # Flat labels/ at same level as images/
    labels_dir = img_path.parent.parent / "labels" if img_path.parent.name in ("train", "val", "valid") else img_path.parent.parent / "labels"
    label_path = labels_dir / img_path.relative_to(img_path.parent).with_suffix(".txt")
    if label_path.exists():
        return label_path
    return None


def _split_from_path(path: Path, root: Path) -> str:
    """Return 'train' or 'val' from path segments under root."""
    rel = path.relative_to(root).as_posix().lower()
    if "val" in rel or "valid" in rel:
        return "val"
    return "train"


def _collect_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """Collect (img_path, label_path, split) under root. split is 'train' or 'val'."""
    root = root.resolve()
    pairs: list[tuple[Path, Path, str]] = []
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        try:
            img_path.relative_to(root)
        except ValueError:
            continue
        label_path = _find_label_for_image(img_path, root)
        if label_path is None:
            continue
        split = _split_from_path(img_path, root)
        pairs.append((img_path, label_path, split))
    return pairs


def _ingest(
    raw_dir: Path,
    out_dir: Path,
    folder_prefix: str,
    train_ratio: float = 0.8,
) -> tuple[int, int, int, int]:
    """
    Ingest from raw_dir subfolders named {folder_prefix}_* into out_dir.
    Returns (total_pairs, train_count, val_count, missing_labels).
    """
    raw_dir = Path(raw_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_images_train = out_dir / "images" / "train"
    out_images_val = out_dir / "images" / "val"
    out_labels_train = out_dir / "labels" / "train"
    out_labels_val = out_dir / "labels" / "val"
    for d in (out_images_train, out_images_val, out_labels_train, out_labels_val):
        d.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        return 0, 0, 0, 0

    all_pairs: list[tuple[Path, Path, str, str]] = []
    missing = 0

    for sub in sorted(raw_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith(folder_prefix + "_"):
            continue
        prefix = sub.name
        pairs = _collect_pairs(sub)
        for img_path, label_path, split in pairs:
            all_pairs.append((img_path, label_path, split, prefix))

    # If no split in source, assign 80/20
    has_val = any(p[2] == "val" for p in all_pairs)
    if all_pairs and not has_val:
        random.shuffle(all_pairs)
        n = len(all_pairs)
        idx = int(n * train_ratio)
        new_all = []
        for i, (img, lbl, _, pre) in enumerate(all_pairs):
            new_all.append((img, lbl, "val" if i >= idx else "train", pre))
        all_pairs = new_all
    elif all_pairs:
        random.shuffle(all_pairs)

    seen = set()
    train_count = 0
    val_count = 0
    for img_path, label_path, split, prefix in all_pairs:
        stem = img_path.stem
        base_name = f"{prefix}_{stem}"
        if base_name in seen:
            continue
        seen.add(base_name)
        img_dest = (out_images_train if split == "train" else out_images_val) / f"{base_name}{img_path.suffix}"
        lbl_dest = (out_labels_train if split == "train" else out_labels_val) / f"{base_name}.txt"
        shutil.copy2(img_path, img_dest)
        shutil.copy2(label_path, lbl_dest)
        if split == "train":
            train_count += 1
        else:
            val_count += 1
    total = train_count + val_count
    return total, train_count, val_count, missing


def _validate_and_report(out_dir: Path, kind: str) -> None:
    """Check that every image has a label and every label has an image; print report."""
    out_dir = Path(out_dir).resolve()
    images_train = set((p.stem for p in (out_dir / "images" / "train").iterdir() if p.suffix.lower() in IMG_EXTENSIONS))
    images_val = set((p.stem for p in (out_dir / "images" / "val").iterdir() if p.suffix.lower() in IMG_EXTENSIONS))
    labels_train = set((p.stem for p in (out_dir / "labels" / "train").glob("*.txt")))
    labels_val = set((p.stem for p in (out_dir / "labels" / "val").glob("*.txt")))
    img_no_lbl_train = images_train - labels_train
    img_no_lbl_val = images_val - labels_val
    lbl_no_img_train = labels_train - images_train
    lbl_no_img_val = labels_val - images_val
    pairs_train = len(images_train & labels_train)
    pairs_val = len(images_val & labels_val)
    print(f"  [{kind}] Valid pairs: train={pairs_train}, val={pairs_val}")
    if img_no_lbl_train or img_no_lbl_val:
        print(f"  [{kind}] Images without label: train={len(img_no_lbl_train)}, val={len(img_no_lbl_val)}")
    if lbl_no_img_train or lbl_no_img_val:
        print(f"  [{kind}] Labels without image: train={len(lbl_no_img_train)}, val={len(lbl_no_img_val)}")


def ingest_camels(raw_dir: str = "raw", out_dir: str = "dataset") -> None:
    """Ingest camel detection data from raw/camel_* into dataset/."""
    raw_dir = Path(raw_dir).resolve()
    out_dir = Path(out_dir).resolve()
    total, train_count, val_count, missing = _ingest(raw_dir, out_dir, "camel")
    print("Camels:")
    print(f"  Total pairs: {total}, train: {train_count}, val: {val_count}, missing labels: {missing}")
    if total > 0:
        _validate_and_report(out_dir, "camel")


def ingest_roads(raw_dir: str = "raw", out_dir: str = "road_dataset") -> None:
    """Ingest road segmentation data from raw/road_* into road_dataset/."""
    raw_dir = Path(raw_dir).resolve()
    out_dir = Path(out_dir).resolve()
    total, train_count, val_count, missing = _ingest(raw_dir, out_dir, "road")
    print("Roads:")
    print(f"  Total pairs: {total}, train: {train_count}, val: {val_count}, missing labels: {missing}")
    if total > 0:
        _validate_and_report(out_dir, "road")


def main() -> None:
    ingest_camels()
    ingest_roads()


if __name__ == "__main__":
    main()
