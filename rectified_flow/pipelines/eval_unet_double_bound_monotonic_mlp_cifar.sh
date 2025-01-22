#!/bin/bash
#SBATCH --job-name=unet_boundary_models
#SBATCH --output=/home/lbo/research/tmp/rectified-flow/evallogs/train_unet_double_bound_monotonic_mlp_cifar.log
#SBATCH --nodes=1
#SBATCH --account=storygen
#SBATCH --qos=storygen_high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50

python /home/lbo/research/tmp/rectified-flow/rectified_flow/pipelines/calc_fid.py --base_dir=/checkpoint/storygen/lbo/boundary_models/train_unet_double_bound_monotonic_mlp_cifar/checkpoint-200000/samples