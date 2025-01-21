#!/bin/bash
#SBATCH --job-name=unet_boundary_models
#SBATCH --output=/home/lbo/research/tmp/rectified-flow/samplelogs/train_unet_double_bound_monotonic_mlp_cifar.log
#SBATCH --nodes=1
#SBATCH --account=storygen
#SBATCH --qos=storygen_high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50

python /home/lbo/research/tmp/rectified-flow/sample_cifar.py --ckpt_path /checkpoint/storygen/lbo/boundary_models/train_unet_double_bound_monotonic_mlp_cifar/checkpoint-200000 --learned_gate 1