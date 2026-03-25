"""
inference.py
------------
Runs a trained YOLOv8 model on the test set and produces a submission CSV
in the format required by the FathomNet 2026 Kaggle competition.

Expected CSV columns (check sample_submission.csv to confirm):
    image_id, category_id, bbox, score
where bbox is a JSON-encoded [x_min, y_min, width, height] list (COCO style).

Usage:
    python scripts/inference.py \
        --weights  outputs/checkpoints/baseline/weights/best.pt \
        --test_ann data/annotations/dataset_test.json \
        --img_dir  data/raw/test \
        --out      outputs/submissions/submission.csv \
        --conf     0.25 \
        --iou      0.5 \
        --img_size 640
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


def load_category_map(data_yaml_dir: str) -> dict[int, int]:
    """
    Load the zero-index -> COCO category_id reverse mapping.
    This is the inverse of what convert_to_yolo.py wrote.
    """
    map_path = Path(data_yaml_dir) / "category_id_map.json"
    with open(map_path) as f:
        data = json.load(f)
    # cat_map was {coco_id: yolo_idx}; we want {yolo_idx: coco_id}
    return {v: int(k) for k, v in data["cat_map"].items()}


def main(args):
    model = YOLO(args.weights)
    model.eval()

    # Load test annotation to get image IDs and filenames
    with open(args.test_ann) as f:
        test_coco = json.load(f)

    img_dir = Path(args.img_dir)
    # Map filename -> image_id
    filename_to_id = {img["file_name"]: img["id"] for img in test_coco["images"]}

    # Load category map (zero-index -> COCO category_id)
    yolo_to_coco = load_category_map(Path(args.weights).parent.parent.parent)

    rows = []
    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"Running inference on {len(image_paths)} images...")

    # Run in batches for speed; tqdm shows per-image progress
    for img_path in tqdm(image_paths, desc="Inference"):
        filename = img_path.name
        image_id = filename_to_id.get(filename)
        if image_id is None:
            # Try without extension mismatch
            for ext in [".jpg", ".jpeg", ".png"]:
                alt = img_path.stem + ext
                image_id = filename_to_id.get(alt)
                if image_id is not None:
                    break
        if image_id is None:
            print(f"  WARNING: no annotation entry for {filename}, skipping")
            continue

        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.img_size,
            verbose=False,
        )

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                # xyxy -> xywh (absolute pixels, COCO style)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
                score = float(boxes.conf[i])
                yolo_cls = int(boxes.cls[i])
                coco_cat_id = yolo_to_coco.get(yolo_cls, yolo_cls)

                rows.append({
                    "image_id":   image_id,
                    "category_id": coco_cat_id,
                    "bbox":       json.dumps([round(x, 2), round(y, 2),
                                              round(w, 2), round(h, 2)]),
                    "score":      round(score, 6),
                })

    df = pd.DataFrame(rows, columns=["image_id", "category_id", "bbox", "score"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Submission written: {out_path}  ({len(df)} detections across {df['image_id'].nunique()} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  required=True, help="Path to best.pt")
    parser.add_argument("--test_ann", default="data/annotations/dataset_test.json")
    parser.add_argument("--img_dir",  default="data/raw/test")
    parser.add_argument("--out",      default="outputs/submissions/submission.csv")
    parser.add_argument("--conf",     type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",      type=float, default=0.5,  help="NMS IoU threshold")
    parser.add_argument("--img_size", type=int,   default=640,  help="Inference image size")
    main(parser.parse_args())
