#!/bin/bash
#SBATCH --job-name=unet_boundary_models
#SBATCH --output=/home/lbo/research/tmp/rectified-flow/evallogs/unet_double_boundary_sin.log
#SBATCH --nodes=1
#SBATCH --account=storygen
#SBATCH --qos=storygen_high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50

python /home/lbo/research/tmp/rectified-flow/rectified_flow/pipelines/calc_fid.py --base_dir=/checkpoint/storygen/lbo/boundary_models/unet_double_boundary_sin/checkpoint-200000/samples