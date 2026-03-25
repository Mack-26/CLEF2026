"""
pseudo_label.py
---------------
Self-training / pseudo-labeling loop for Positive-Unlabeled (PU) detection.

STATUS: STUB — to be implemented as the class project contribution.

The algorithm this will implement:
  Round 0:  Train baseline on labeled data only  (src/trainer.py)
  Round k+1:
    1. Run Round-k model over all UNLABELED training images
    2. Keep detections above `conf_threshold` as pseudo-labels
    3. Merge pseudo-labels with original ground-truth annotations
    4. Re-train model from scratch (or fine-tune) on the merged set
    5. Repeat until convergence or max_rounds

Key design decisions to make:
  - Confidence threshold schedule (fixed vs. adaptive per class)
  - Whether to re-train from scratch or fine-tune each round
  - How to handle conflicting pseudo-labels vs. GT annotations

References:
  - Rizve et al., "In Defense of Pseudo-Labels" (ICLR 2021)
  - Liu et al., "Unbiased Teacher for Semi-Supervised Object Detection" (ICLR 2021)
"""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

from src.trainer import run_training


def get_unlabeled_image_ids(
    train_ann_path: str,
    labeled_threshold: int = 1,
) -> list[int]:
    """
    Return image IDs that have fewer than `labeled_threshold` annotations.

    In a PU setting, these are images that may contain positive instances
    that were deliberately left unannotated during dataset curation.

    Parameters
    ----------
    train_ann_path : str
        Path to dataset_train.json.
    labeled_threshold : int
        Images with fewer annotations than this count are treated as unlabeled.
        Set to 1 to treat 0-annotation images as unlabeled.
    """
    with open(train_ann_path) as f:
        coco = json.load(f)

    ann_counts: dict[int, int] = {img["id"]: 0 for img in coco["images"]}
    for ann in coco["annotations"]:
        ann_counts[ann["image_id"]] = ann_counts.get(ann["image_id"], 0) + 1

    return [img_id for img_id, count in ann_counts.items() if count < labeled_threshold]


def generate_pseudo_labels(
    model_weights: str,
    unlabeled_img_dir: str,
    conf_threshold: float = 0.5,
    iou_threshold: float  = 0.5,
    img_size: int         = 640,
) -> list[dict]:
    """
    Run the model over unlabeled images and return pseudo-annotation dicts
    in COCO annotation format.

    TODO: implement using ultralytics YOLO.predict()
    """
    raise NotImplementedError(
        "pseudo-label generation not yet implemented — see class project contribution"
    )


def merge_annotations(
    original_ann_path: str,
    pseudo_labels: list[dict],
    out_ann_path: str,
) -> None:
    """
    Append pseudo_labels into a copy of the original COCO JSON and write it
    to out_ann_path. Used to build the combined dataset for the next round.

    TODO: implement — be careful about unique annotation IDs.
    """
    raise NotImplementedError


def run_pseudo_label_loop(
    cfg: dict,
    train_ann_path: str,
    unlabeled_img_dir: str,
    max_rounds: int    = 3,
    conf_threshold: float = 0.5,
) -> str:
    """
    Outer loop: baseline → pseudo-label → retrain → ...

    Parameters
    ----------
    cfg : dict
        Training config (loaded from configs/train_config.yaml).
    train_ann_path : str
        Path to original dataset_train.json.
    unlabeled_img_dir : str
        Directory containing the unlabeled training images.
    max_rounds : int
        Maximum number of pseudo-labeling rounds.
    conf_threshold : float
        Detection confidence required to accept a pseudo-label.

    Returns
    -------
    str
        Path to the best weights after the final round.
    """
    # Round 0: baseline on labeled data
    print("=== Round 0: Baseline training ===")
    best_weights = run_training(cfg)

    for round_num in range(1, max_rounds + 1):
        print(f"\n=== Round {round_num}: Pseudo-labeling ===")

        pseudo_labels = generate_pseudo_labels(
            model_weights=best_weights,
            unlabeled_img_dir=unlabeled_img_dir,
            conf_threshold=conf_threshold,
        )
        print(f"  Generated {len(pseudo_labels)} pseudo-annotations")

        # Build a new annotation file with pseudo-labels merged in
        round_ann = f"data/annotations/dataset_train_round{round_num}.json"
        merge_annotations(train_ann_path, pseudo_labels, round_ann)

        # TODO: call convert_to_yolo.py on the new annotation file to build
        # a fresh YOLO dataset, then pass its YAML to run_training below.
        round_yaml = f"data/yolo_round{round_num}/dataset.yaml"

        # Update run name so outputs don't overwrite each other
        round_cfg = deepcopy(cfg)
        round_cfg["output"]["name"] = f"{cfg['output']['name']}_round{round_num}"

        best_weights = run_training(round_cfg, dataset_yaml=round_yaml)

    return best_weights
