"""
convert_to_yolo.py
------------------
Converts COCO-format annotations to the YOLO flat-file format expected by
Ultralytics YOLOv8.

YOLO label format (one .txt per image, same stem as the image file):
    <class_id> <cx> <cy> <w> <h>
All values are normalized to [0, 1] by the image dimensions.

This script also:
  - Writes a dataset YAML file that YOLOv8 reads at training time
  - Optionally splits the training set into train/val splits

Usage:
    python scripts/convert_to_yolo.py \
        --train_ann data/annotations/dataset_train.json \
        --test_ann  data/annotations/dataset_test.json \
        --img_dir   data/raw \
        --out_dir   data/yolo \
        --val_frac  0.1
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm


def load_coco(ann_path: str) -> dict:
    with open(ann_path) as f:
        return json.load(f)


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox: [x_min, y_min, width, height]  (absolute pixels)
    YOLO bbox: [cx, cy, w, h]                 (normalized 0–1)
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1] to handle any annotation overflow
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def write_split(
    image_ids: list[int],
    img_info: dict,        # image_id -> image dict
    ann_by_img: dict,      # image_id -> list of annotation dicts
    cat_map: dict,         # COCO category_id -> zero-indexed class_id
    src_img_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    split_name: str,
):
    """Write images and label files for one split (train/val/test)."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for img_id in tqdm(image_ids, desc=f"  {split_name}"):
        info = img_info[img_id]
        filename = info["file_name"]
        img_w = info["width"]
        img_h = info["height"]

        # Locate source image (search train/ and test/ subdirs)
        src = None
        for subdir in ["train", "test", ""]:
            candidate = src_img_dir / subdir / filename
            if candidate.exists():
                src = candidate
                break
        if src is None:
            # Try just the filename directly
            candidate = src_img_dir / filename
            if candidate.exists():
                src = candidate

        if src is None:
            skipped += 1
            continue

        # Hard-link (fast, no extra disk usage) or copy
        dst_img = dst_img_dir / filename
        if not dst_img.exists():
            try:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst_img)
            except Exception:
                shutil.copy2(src, dst_img)

        # Write label file
        anns = ann_by_img.get(img_id, [])
        lbl_path = dst_lbl_dir / (Path(filename).stem + ".txt")
        with open(lbl_path, "w") as f:
            for ann in anns:
                cls = cat_map[ann["category_id"]]
                cx, cy, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        # Images with no annotations get an empty label file (YOLOv8 handles this)

    if skipped:
        print(f"  WARNING: {skipped} images not found on disk for split '{split_name}'")


def main(args):
    random.seed(42)

    print("Loading annotations...")
    train_coco = load_coco(args.train_ann)
    test_coco  = load_coco(args.test_ann)

    # Build category mapping: COCO IDs are not necessarily 0-indexed
    # Sort by id for determinism; map to 0, 1, 2, ...
    categories = sorted(train_coco["categories"], key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]
    print(f"Categories ({len(class_names)}): {class_names}")

    # Index images and annotations
    train_img_info = {img["id"]: img for img in train_coco["images"]}
    test_img_info  = {img["id"]: img for img in test_coco["images"]}

    train_ann_by_img: dict[int, list] = defaultdict(list)
    for ann in train_coco["annotations"]:
        # Skip crowd annotations (area=0 or iscrowd=1)
        if ann.get("iscrowd", 0):
            continue
        train_ann_by_img[ann["image_id"]].append(ann)

    # Train / val split
    all_train_ids = list(train_img_info.keys())
    random.shuffle(all_train_ids)
    n_val = max(1, int(len(all_train_ids) * args.val_frac))
    val_ids   = all_train_ids[:n_val]
    train_ids = all_train_ids[n_val:]
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    out = Path(args.out_dir)
    src = Path(args.img_dir)

    # Write each split
    for split_name, ids, img_info, ann_by_img in [
        ("train", train_ids, train_img_info, train_ann_by_img),
        ("val",   val_ids,   train_img_info, train_ann_by_img),
        ("test",  [img["id"] for img in test_coco["images"]], test_img_info, {}),
    ]:
        write_split(
            ids, img_info, ann_by_img, cat_map, src,
            dst_img_dir=out / "images" / split_name,
            dst_lbl_dir=out / "labels" / split_name,
            split_name=split_name,
        )

    # Write dataset YAML for YOLOv8
    dataset_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = Path(args.out_dir) / "dataset.yaml"
    # yaml.dump doesn't format lists nicely; write manually for readability
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"Dataset YAML written to {yaml_path}")

    # Save the category mapping for use during inference
    map_path = Path(args.out_dir) / "category_id_map.json"
    with open(map_path, "w") as f:
        json.dump({"cat_map": {str(k): v for k, v in cat_map.items()},
                   "names": class_names}, f, indent=2)
    print(f"Category map written to {map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ann", default="data/annotations/train_dataset.json")
    parser.add_argument("--test_ann",  default="data/annotations/test_dataset.json")
    parser.add_argument("--img_dir",   default="data/raw")
    parser.add_argument("--out_dir",   default="data/yolo")
    parser.add_argument("--val_frac",  type=float, default=0.1,
                        help="Fraction of training images to hold out for validation")
    main(parser.parse_args())
