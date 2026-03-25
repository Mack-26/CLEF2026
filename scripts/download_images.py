"""
download_images.py
------------------
Robust image downloader for the FathomNet 2026 dataset.

Images are specified by `coco_url` fields in the COCO-format annotation JSON.
This script:
  - Skips images that already exist on disk
  - Retries transient failures with exponential backoff
  - Downloads in parallel using a thread pool
  - Logs all failures to a file so you can re-run only the bad ones

Usage:
    python scripts/download_images.py \
        --ann data/annotations/dataset_train.json \
        --out data/raw/train \
        --workers 8

    python scripts/download_images.py \
        --ann data/annotations/dataset_test.json \
        --out data/raw/test \
        --workers 8
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_RETRIES = 5          # total attempts per image
BACKOFF_BASE = 2.0       # seconds; sleep = BACKOFF_BASE ** attempt
TIMEOUT = 30             # seconds per HTTP request
CHUNK_SIZE = 1 << 16     # 64 KB write chunks


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("downloader")
    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))

    # File handler — everything, including per-image failures
    fh = logging.FileHandler(log_dir / "download_failures.log")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def download_one(url: str, dest: Path, logger: logging.Logger) -> tuple[str, bool]:
    """
    Download a single image with retries.
    Returns (url, success).
    """
    if dest.exists():
        return url, True   # already downloaded — skip

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    f.write(chunk)
            tmp.rename(dest)
            return url, True
        except Exception as exc:
            wait = BACKOFF_BASE ** attempt
            logger.debug(f"Attempt {attempt+1}/{MAX_RETRIES} failed for {url}: {exc}. Retrying in {wait:.1f}s")
            time.sleep(wait)
            if tmp.exists():
                tmp.unlink()

    logger.warning(f"FAILED after {MAX_RETRIES} attempts: {url}")
    return url, False


def main(args):
    logger = setup_logging(Path("outputs/logs"))

    # Load annotations
    ann_path = Path(args.ann)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Found {len(images)} images in {ann_path.name}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Workers: {args.workers}")

    # Build (url, dest_path) pairs
    # FathomNet COCO JSON stores the download URL in `coco_url`.
    # Fall back to `url` if `coco_url` is absent.
    tasks = []
    for img in images:
        url = img.get("coco_url") or img.get("url") or ""
        if not url:
            logger.warning(f"No URL for image id={img['id']} file={img.get('file_name')}; skipping")
            continue
        filename = img.get("file_name") or Path(url).name
        dest = out_dir / filename
        tasks.append((url, dest))

    already_done = sum(1 for _, dest in tasks if dest.exists())
    logger.info(f"Already downloaded: {already_done}/{len(tasks)}")

    # Download in parallel
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, url, dest, logger): url for url, dest in tasks}
        with tqdm(total=len(tasks), desc="Downloading", unit="img") as pbar:
            for fut in as_completed(futures):
                url, ok = fut.result()
                if not ok:
                    failed.append(url)
                pbar.update(1)

    logger.info(f"Done. Failed: {len(failed)}/{len(tasks)}")
    if failed:
        fail_file = Path("outputs/logs") / f"failed_{ann_path.stem}.txt"
        with open(fail_file, "w") as f:
            f.write("\n".join(failed))
        logger.warning(f"Failed URLs written to {fail_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FathomNet dataset images")
    parser.add_argument("--ann", required=True, help="Path to COCO annotation JSON")
    parser.add_argument("--out", required=True, help="Output directory for images")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download threads")
    main(parser.parse_args())
