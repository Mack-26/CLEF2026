#!/bin/bash
# train_baseline.sh
# -----------------
# SLURM job script for training the FathomNet baseline on Great Lakes (UMich).
#
# Submit with:
#   sbatch slurm/train_baseline.sh
#
# Check status:
#   squeue -u $USER
#
# View output live:
#   tail -f outputs/logs/train_baseline_<JOBID>.out

# ---- Resource requests -------------------------------------------------------
#SBATCH --job-name=fathomnet_baseline
#SBATCH --account=engin1
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/train_baseline_%j.out
#SBATCH --error=outputs/logs/train_baseline_%j.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=aromanan@umich.edu

# ---- Environment setup -------------------------------------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
date

# Load modules available on Great Lakes
module load python3.11-anaconda/2024.02
module load cuda/12.1

# Activate your virtual environment
# (create once with: python -m venv ~/venvs/fathomnet && pip install -r requirements.txt)
source ~/venvs/fathomnet/bin/activate

# Move to project root
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# ---- Convert annotations to YOLO format (fast; always safe to re-run) -------
echo "Converting annotations to YOLO format..."
python scripts/convert_to_yolo.py \
    --train_ann data/annotations/train_dataset.json \
    --test_ann  data/annotations/test_dataset.json \
    --img_dir   data/raw \
    --out_dir   data/yolo \
    --val_frac  0.1

# ---- Train -------------------------------------------------------------------
echo "Starting training..."
python src/trainer.py --config configs/train_config.yaml

# ---- Inference + submission --------------------------------------------------
echo "Running inference on test set..."
python scripts/inference.py \
    --weights  outputs/checkpoints/baseline/weights/best.pt \
    --test_ann data/annotations/test_dataset.json \
    --img_dir  data/raw/test \
    --out      outputs/submissions/submission_${SLURM_JOB_ID}.csv \
    --conf     0.25 \
    --iou      0.5

echo "Done at $(date)"
