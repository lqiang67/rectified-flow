#!/bin/bash
#SBATCH --job-name=unet_boundary_models
#SBATCH --output=/home/lbo/research/tmp/rectified-flow/samplelogs/unet_double_boundary_sin.log
#SBATCH --nodes=1
#SBATCH --account=storygen
#SBATCH --qos=storygen_high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50

python /home/lbo/research/tmp/rectified-flow/sample_cifar.py --ckpt_path /checkpoint/storygen/lbo/boundary_models/unet_double_boundary_sin/checkpoint-200000 --learned_gate 0