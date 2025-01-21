import argparse
import torch
import os
import numpy as np
from torchvision.utils import save_image

from rectified_flow.samplers import (
    EulerSampler,
    SDESampler,
    OverShootingSampler,
    StochasticCurvedEulerSampler
)
from rectified_flow import utils
from rectified_flow import RectifiedFlow
from rectified_flow.models.unet_bound import SongUNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

seed = 0
device = "cuda"

batch_size = 500  # number of samples each iteration
iterations = 20  # batch_size * iterations = total number of samples
#num_steps_list = [16, 32, 64, 128, 256, 512, 1024]
num_steps_list = [256, 512, 1024]


def sample_loop(sampler, current_output_dir):
    for iteration in range(iterations):
        print(f"Iteration {iteration}, generating batch from {iteration * batch_size} to {(iteration + 1) * batch_size}")
        z_1 = sampler.sample_loop(seed=iteration).trajectories[-1]
        z_1 = (z_1 * 0.5 + 0.5).clamp(0, 1)
        for i in range(z_1.shape[0]):
            save_image(
                z_1[i],
                os.path.join(current_output_dir, f"{iteration}_{i}.png")
            )


def create_sampler(sampler_class, rectified_flow, **kwargs):
    return sampler_class(rectified_flow=rectified_flow, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='RectifiedFlow Sampler')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to checkpoint directory, e.g. /root/autodl-tmp/unet_cifar_double_bound/checkpoint-500000')
    parser.add_argument('--learned_gate', type=int, default=0)
    args = parser.parse_args()

    # 输出目录
    ckpt_path = args.ckpt_path
    output_dir = os.path.join(ckpt_path, "samples")

    utils.set_seed(seed)

    # 修改此处 data_mean 文件位置
    data_mean = torch.load(
        "/home/lbo/research/tmp/my_boundary/cifar10_mean.pt"
    ).to(device)

    if args.learned_gate > 0:
        # ======== double bound model with path mlp, uncomment this block ========
        from rectified_flow.models.unet_double_bound_monotonic_mlp import DoubleBoundMonotonicMLP, SongUNetConfig
        UNet_config = SongUNetConfig(
                img_resolution=32,
                in_channels=3,
                out_channels=3,
            )
        model = DoubleBoundMonotonicMLP(UNet_config, data_mean)
        state_dict = torch.load(f"{ckpt_path}/unet_ema.pt")
        model.load_state_dict(state_dict)
    else:
        # ======== Vanilla model, single boundary model, and double boundary model ========
        model = SongUNet.from_pretrained(
            ckpt_path,
            use_ema=True,
            data_mean=data_mean
        )

    model = model.to(device)
    model.eval()

    rectified_flow = RectifiedFlow(
        data_shape=(3, 32, 32),
        velocity_field=model,
        device=device,
    )

    print(f"Base output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    sampler_configs = []
    for num_steps in num_steps_list:
        # (A) Euler Sampler
        sampler_configs.append({
            "sampler_class": EulerSampler,
            "kwargs": {
                "num_steps": num_steps,
                "num_samples": batch_size,
            },
            "subdir_name": f"EulerSampler_steps{num_steps}"
        })

        # (B) DDPM Sampler (StochasticCurvedEulerSampler with noise_replacement_rate="ddpm")
        sampler_configs.append({
            "sampler_class": StochasticCurvedEulerSampler,
            "kwargs": {
                "num_steps": num_steps,
                "num_samples": batch_size,
                "noise_replacement_rate": "ddpm",
            },
            "subdir_name": f"DDPMSampler_steps{num_steps}"
        })

        # (C) OverShooting Sampler, 对多个 c 参数分别采样
        for c in [1, 2, 5]:
            sampler_configs.append({
                "sampler_class": OverShootingSampler,
                "kwargs": {
                    "num_steps": num_steps,
                    "num_samples": batch_size,
                    "c": c,
                },
                "subdir_name": f"OverShootingSampler_steps{num_steps}_c{c}"
            })

        # (D) SDE Sampler without noise_decay
        for noise_scale in [1., 2., 3.]:
            sampler_configs.append({
                "sampler_class": SDESampler,
                "kwargs": {
                    "num_steps": num_steps,
                    "num_samples": batch_size,
                    "noise_scale": noise_scale,
                    "noise_decay_rate": 0.,
                    #"score_scale": 1.0,
                },
                "subdir_name": f"SDESampler_steps{num_steps}_noise_scale{noise_scale}_noise_decay_rate_{0.}"
            })

        # (E) SDE Sampler with noise_decay
        for noise_scale in [1., 3., 5.]:
            for noise_decay_rate in [1.]:
                sampler_configs.append({
                    "sampler_class": SDESampler,
                    "kwargs": {
                        "num_steps": num_steps,
                        "num_samples": batch_size,
                        "noise_scale": noise_scale,
                        "noise_decay_rate": noise_decay_rate,
                        #"score_scale": 1.0,
                    },
                    "subdir_name": f"SDESampler_steps{num_steps}_noise_scale{noise_scale}_noise_decay_rate{noise_decay_rate}"
                })

    with torch.inference_mode():
        for config in sampler_configs:
            print(f"Sampling with {config['subdir_name']}")
            sampler_class = config["sampler_class"]
            subdir_name = config["subdir_name"]
            sampler_kwargs = config["kwargs"]

            current_output_dir = os.path.join(output_dir, subdir_name)
            os.makedirs(current_output_dir, exist_ok=True)

            sampler = create_sampler(
                sampler_class=sampler_class,
                rectified_flow=rectified_flow,
                **sampler_kwargs
            )

            sample_loop(sampler, current_output_dir)


if __name__ == "__main__":
    main()
