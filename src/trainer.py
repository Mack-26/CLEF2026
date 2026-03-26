"""
trainer.py
----------
Baseline training pipeline using YOLOv8 (Ultralytics).

This module is intentionally kept thin: it parses a config dict and calls
Ultralytics' trainer. All the heavy lifting (augmentation, mixed precision,
distributed training) is handled by the library.

The pseudo-labeling extension will import `run_training` and call it with an
updated dataset YAML that contains extra pseudo-labeled images, so the
interface is designed to be reused with minimal changes.

Usage (direct):
    python src/trainer.py --config configs/train_config.yaml

Usage (from another module, e.g., pseudo-labeling loop):
    from src.trainer import run_training
    run_training(cfg, dataset_yaml="data/yolo_pseudo/dataset.yaml")
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def run_training(cfg: dict, dataset_yaml: str | None = None) -> str:
    """
    Train a YOLOv8 model and return the path to the best checkpoint.

    Parameters
    ----------
    cfg : dict
        Full config dict (see configs/train_config.yaml for keys).
    dataset_yaml : str, optional
        Override the dataset YAML path from cfg (useful for pseudo-label rounds).

    Returns
    -------
    str
        Absolute path to the best checkpoint weights (best.pt).
    """
    data_yaml = dataset_yaml or cfg["dataset"]["yaml"]
    model_cfg  = cfg["model"]["architecture"]  # e.g. "yolov8x.pt" or a .yaml
    project    = cfg["output"]["project"]
    run_name   = cfg["output"]["name"]

    # Resume from last.pt if it exists (handles SLURM timeout restarts)
    last_weights = Path(project) / run_name / "weights" / "last.pt"
    if last_weights.exists():
        print(f"Resuming from checkpoint: {last_weights}")
        model = YOLO(str(last_weights))
        resume = True
    else:
        print(f"Starting fresh training with: {model_cfg}")
        model = YOLO(model_cfg)
        resume = False

    results = model.train(
        data=data_yaml,
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["training"]["img_size"],
        batch=cfg["training"]["batch_size"],
        lr0=cfg["training"]["lr0"],
        lrf=cfg["training"]["lrf"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_epochs=cfg["training"]["warmup_epochs"],
        # Augmentation
        hsv_h=cfg["augmentation"]["hsv_h"],
        hsv_s=cfg["augmentation"]["hsv_s"],
        hsv_v=cfg["augmentation"]["hsv_v"],
        flipud=cfg["augmentation"]["flipud"],
        fliplr=cfg["augmentation"]["fliplr"],
        mosaic=cfg["augmentation"]["mosaic"],
        # Output
        project=project,
        name=run_name,
        exist_ok=True,         # don't rename the run dir on restart
        save=True,
        save_period=cfg["output"].get("save_period", -1),
        resume=resume,         # picks up epoch count, optimizer state, LR schedule
        # Misc
        workers=cfg["training"].get("workers", 8),
        device=cfg["training"].get("device", 0),
        amp=cfg["training"].get("amp", True),
        verbose=True,
    )

    best_weights = Path(project) / run_name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return str(best_weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument(
        "--dataset_yaml",
        default=None,
        help="Override dataset YAML (useful for pseudo-label rounds)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_training(cfg, dataset_yaml=args.dataset_yaml)


if __name__ == "__main__":
    main()
