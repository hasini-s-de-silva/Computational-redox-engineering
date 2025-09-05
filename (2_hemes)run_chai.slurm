#!/bin/bash
#SBATCH --job-name=chai_two_hemes
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=chai%j.out
#SBATCH --error=chai%j.err
#SBATCH --time=3-00:00:00
#SBATCH --account=bisc033004
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hasinibristol@gmail.com

# Load necessary modules (adjust if needed)
module load cuda/12.4.1-ebv6

# Activate Python virtual environment
source ~/chai-env/bin/activate

# Navigate to your project directory
cd "/user/work/bx24552/FINAL/2_hemes"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the updated Python script
python run_chai_cytbx.py