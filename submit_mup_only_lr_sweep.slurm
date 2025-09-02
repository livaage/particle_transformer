#!/bin/bash
#SBATCH --job-name=mup_only_lr_sweep
#SBATCH --output=logs/mup_only_lr_sweep_%j.out
#SBATCH --error=logs/mup_only_lr_sweep_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the micromamba environment
source ~/.bashrc
micromamba activate particle_transformer

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "Starting μP Learning Rate Sweep (μP Models Only)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# Run the μP-only learning rate sweep
python run_mup_only_lr_sweep.py

echo "μP learning rate sweep completed at $(date)"
