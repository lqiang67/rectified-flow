#!/bin/bash
#SBATCH --job-name=unet_boundary_models
#SBATCH --output=/home/lbo/research/tmp/rectified-flow/trainlogs/unet_double_bound_monotonic_mlp_sin.log
#SBATCH --nodes=1
#SBATCH --account=storygen
#SBATCH --qos=storygen_high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=50

export OUTPUT_DIR="/checkpoint/storygen/lbo/boundary_models/unet_double_bound_monotonic_mlp_sin"
export DATA_ROOT="/home/lbo/research/tmp/rectified-flow/data/cifar10"

accelerate launch --multi_gpu --num_processes=8 --main_process_port=12353 -m rectified_flow.pipelines.train_unet_double_bound_monotonic_mlp_cifar \
  --output_dir="$OUTPUT_DIR" \
  --resume_from_checkpoint="latest" \
  --data_root="$DATA_ROOT" \
  --seed=0 \
  --resolution=32 \
  --train_batch_size=128 \
  --max_train_steps=200000 \
  --checkpointing_steps=50000 \
  --learning_rate=0.0016 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=10000 \
  --random_flip \
  --allow_tf32 \
  --interp="straight" \
  --source_distribution="normal" \
  --is_independent_coupling=True \
  --train_time_distribution="lognormal" \
  --train_time_weight="uniform" \
  --criterion="mse" \
  --use_ema