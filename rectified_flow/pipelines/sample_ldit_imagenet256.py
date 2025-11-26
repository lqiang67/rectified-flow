#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accelerate-style LightningDiT sampling
- bf16 autocast + optional torch.compile (via Accelerate TorchDynamoPlugin)
- Fixed latents per-eval (class-balanced) for fair comparison
- Euler sampler only (minimal path)
- Optional metrics via torch-fidelity:
    * Pass 1: FID/IS using fid_statistics_file
    * Pass 2: Precision/Recall using reference 10k images folder (input1=ref, input2=generated)

Notes
-----
* Pass multiple values with spaces where supported, e.g.:
    --nfe-list 64 128 --cfg-scales 1.0 1.5 --cfg-schedules constant linear interval
* Linear schedule uses progress∈[0,1] (model will default to 1 - t if not provided)
* Interval schedule requires pairs in [0,1], e.g.:
    --cfg-schedules interval --cfg-intervals 0.2 0.6 0.8 0.9
  (inside intervals apply `cfg`; outside use pure conditional, i.e., scale=1.)
* By default we use EMA weights for model parameters (disable with --no-ema).
* Default VAE is MAR (KL-16). For SD VAE, use --vae-impl sd and --vae mse/ema.
* For PRC, set --real-images-path to the 10k reference folder
  (e.g. from https://github.com/openai/guided-diffusion/tree/main/evaluations).
"""
from __future__ import annotations

import os
import gc
import json
import math
import time
import argparse
import shutil
from collections import OrderedDict
from typing import List, Sequence, Tuple, Optional, Dict, Any, Callable

import torch
import torch.nn as nn
import numpy as np
from PIL import Image  # noqa: F401
from torchvision.utils import save_image

from rectified_flow.models.lightningdit import LightningDiT_models
from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers.euler_sampler import EulerSampler

from diffusers.models import AutoencoderKL as SDAutoencoderKL
from diffusers.utils.torch_utils import is_compiled_module

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, TorchDynamoPlugin

import torch_fidelity
from tqdm.auto import tqdm


# ---------------------------
# Small utilities
# ---------------------------

def _dedupe_keep_order(seq: Sequence) -> List:
    """De-duplicate while preserving order."""
    seen = set()
    out: List = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _format_float_compact(v: float | None) -> str:
    if v is None:
        return "none"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _format_cfg(v: float | None) -> str:
    return _format_float_compact(v)


def _format_intervals(pairs: Optional[List[Tuple[float, float]]]) -> str:
    if not pairs:
        return "none"
    return "_".join([f"{_format_float_compact(lo)}-{_format_float_compact(hi)}" for lo, hi in pairs])


def _sched_tag(name: str, pairs: Optional[List[Tuple[float, float]]]) -> str:
    name = name.lower()
    if name == "constant":
        return "cfg-const"
    if name == "linear":
        return "cfg-linear"
    if name == "interval":
        return f"cfg-interv_{_format_intervals(pairs)}"
    return f"cfg-{name}"


def _get_local_world_size_fallback() -> int:
    # Prefer env hints (e.g., Slurm), otherwise CUDA count
    cand = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("SLURM_GPUS_ON_NODE")
    if cand:
        try:
            token = cand.split("(")[0].split(",")[0]
            return max(1, int(token))
        except Exception:
            pass
    return max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)


# ---------------------------
# EMA helpers
# ---------------------------
@torch.no_grad()
def _apply_ema_inplace_from_state_dict(
    unwrapped: nn.Module, ema_state_cpu: "OrderedDict[str, torch.Tensor]"
):
    named = dict(unwrapped.named_parameters())
    missing, unexpected = [], []

    for name, p in named.items():
        if name in ema_state_cpu:
            src = ema_state_cpu[name]
            p.data.copy_(src.to(device=p.device, dtype=p.dtype, non_blocking=True))
        else:
            missing.append(name)

    for name in ema_state_cpu.keys():
        if name not in named:
            unexpected.append(name)

    return missing, unexpected


def _unwrap_for_ema(accelerator: Accelerator, m: nn.Module) -> nn.Module:
    u = accelerator.unwrap_model(m)
    return getattr(u, "_orig_mod", u) if is_compiled_module(u) else u


# ---------------------------
# VAE decode construction
# ---------------------------

def build_vae_decode(args, device):
    if args.vae_impl == "sd":
        VAE_SCALE = 0.18215
        vae = SDAutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
        vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

        def _dec(z):  # z: float32
            return vae.decode(z.float() / VAE_SCALE).sample

        decode = _dec
        if args.compile_vae:
            decode = torch.compile(_dec, backend=args.compile_backend, mode=args.compile_mode)
    else:
        # MAR KL-16-like VAE
        try:
            from rectified_flow.models.vae import AutoencoderKL as MARAutoencoderKL  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Cannot find MAR VAE, please ensure rectified_flow.models.vae is available."
            ) from e

        VAE_SCALE = 0.2325
        vae = MARAutoencoderKL(
            embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path
        ).to(device).eval()
        for p in vae.parameters():
            p.requires_grad = False

        def _dec(z):
            return vae.decode(z.float() / VAE_SCALE)

        decode = _dec
        if args.compile_vae:
            decode = torch.compile(_dec, backend=args.compile_backend, mode=args.compile_mode)

    return decode


# ---------------------------
# Arg parser
# ---------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description=(
            "Accelerate-style LDiT sampling with bf16 + optional torch.compile "
            "(EMA tensor-only, fixed eval latents, multi-GPU)"
        )
    )

    # Model / checkpoint paths
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing {filename_base}_model.pt / {filename_base}_ema.pt",
    )
    p.add_argument("--filename-base", type=str, default="ldit")
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Directly specify model .pt path (overrides checkpoint-dir)",
    )
    p.add_argument(
        "--ema-path",
        type=str,
        default="",
        help="Directly specify EMA .pt path (overrides checkpoint-dir)",
    )
    p.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        default=True,
        help="Whether to use EMA weights for sampling (enabled by default)",
    )
    p.add_argument("--no-ema", dest="use_ema", action="store_false", help="Disable EMA weights")

    # LDiT model structure
    p.add_argument("--model", type=str, choices=sorted(list(LightningDiT_models.keys())), default="LightningDiT-XL")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--patch-size", type=int, default=1)
    p.add_argument("--time-scale", type=float, default=1.0)

    # VAE
    p.add_argument("--vae-impl", type=str, default="mar", choices=["sd", "mar"])
    p.add_argument("--vae", type=str, default="ema", choices=["mse", "ema"], help="sd-vae-ft-<vae>")
    p.add_argument(
        "--vae-path",
        type=str,
        default="./pretrained_models/vae/kl16.ckpt",
    )
    p.add_argument("--vae-embed-dim", type=int, default=16)
    p.add_argument("--vae-stride", type=int, default=16)
    p.add_argument("--compile-vae", action="store_true", default=True)

    # Sampling: schedules × nfe × cfg (Euler only)
    p.add_argument(
        "--cfg-schedules",
        nargs="+",
        type=str,
        default=["constant"],
        choices=["constant", "linear", "interval"],
        help="CFG schedule(s): constant | linear | interval",
    )
    p.add_argument(
        "--nfe-list",
        dest="nfe_list",
        nargs="+",
        type=int,
        default=[16, 32, 64],
        metavar="N",
        help="Space-separated list: e.g. --nfe-list 64 128",
    )
    p.add_argument(
        "--cfg-scales",
        dest="cfg_scales",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0],
        metavar="S",
        help="Space-separated list: e.g. --cfg-scales 1.0 1.5",
    )
    p.add_argument(
        "--cfg-intervals",
        dest="cfg_intervals",
        nargs="+",
        type=float,
        default=[],
        help="For 'interval' schedule: give lo hi pairs in [0,1], e.g. --cfg-intervals 0.2 0.6 0.8 0.9",
    )

    p.add_argument("--num-images", type=int, default=10_000, help="Total number of generated images (aligned to num-classes)")
    p.add_argument("--eval-batch-size", type=int, default=200, help="Per-GPU batch size for sampling")
    p.add_argument("--eval-latent-seed", type=int, default=2333, help="Fixed random seed for x0_world")
    p.add_argument("--global-seed", type=int, default=0, help="Global random seed")

    # Evaluation
    p.add_argument(
        "--fid-stats-path",
        type=str,
        default="./fid_stats/adm_in256_stats.npz",
        help="Path to FID statistics (.npz) for FID calculation",
    )
    p.add_argument(
        "--real-images-path",
        type=str,
        default="./data/ref_images_imnet256",
        help="Directory of 10k reference real images for PRC (input1). Generated images folder (input2) is auto-set.",
    )

    # Output
    p.add_argument("--output-dir", type=str, default="./samples", help="Root output directory")

    # Cleanup option
    p.add_argument(
        "--cleanup-images",
        action="store_true",
        default=True,
        help="After metrics are computed, delete the generated images folder to save disk space.",
    )

    # Accelerate / precision / compile / TF32
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow-tf32", action="store_true", default=True)
    p.add_argument("--torch-compile", action="store_true", default=True)
    p.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        choices=["inductor", "aot_eager", "eager"],
        help="Backend for torch.compile (torch_tensorrt may not be supported)",
    )
    p.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])

    return p


# ---------------------------
# Main
# ---------------------------

def main(args):
    # Accelerator / compile / precision
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    logging_dir = os.path.join(args.output_dir, "logs", timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    dynamo = None
    if args.torch_compile:
        dynamo = TorchDynamoPlugin(
            backend=args.compile_backend,
            mode=args.compile_mode,
            fullgraph=False,
        )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,  # "bf16" by default
        dynamo_plugin=dynamo,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    device = accelerator.device
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    world = accelerator.num_processes
    global_rank = accelerator.process_index
    local_rank = accelerator.local_process_index
    local_world_size = _get_local_world_size_fallback()
    num_nodes = max(1, world // max(1, local_world_size))
    accelerator.print(
        f"[dist] rank={global_rank}, local_rank={local_rank}, world_size={world}, "
        f"num_nodes={num_nodes}, gpus_per_node={local_world_size}"
    )

    set_seed(args.global_seed, device_specific=True)

    # Validate time-window args
    for lo, hi, tag in [
        (args.ema_start_time, args.ema_end_time, "EMA"),
    ]:
        assert 0.0 <= float(lo) < float(hi) <= 1.0, f"{tag} window must satisfy 0<=start<end<=1, got ({lo},{hi})"
    assert float(args.ema_off_guidance_scale) >= 0.0, "ema-off-guidance-scale must be >= 0"

    # VAE / latent size
    if args.vae_impl == "sd":
        vae_stride = args.vae_stride or 8
        vae_embed_dim = args.vae_embed_dim or 4
    else:
        vae_stride = args.vae_stride or 16
        vae_embed_dim = args.vae_embed_dim or 16

    assert args.image_size % vae_stride == 0, f"Image size must be divisible by VAE stride={vae_stride}"
    latent_size = args.image_size // vae_stride

    # Model
    base_model = LightningDiT_models[args.model](
        input_size=latent_size,
        patch_size=args.patch_size,
        in_channels=vae_embed_dim,
        num_classes=args.num_classes,
        class_dropout_prob=0.1,  # enable null-class for CFG
        time_scale=args.time_scale,
        use_qknorm=True,
    )

    # Accelerate wrapping (including optional torch.compile)
    model = accelerator.prepare(base_model)
    unwrapped = _unwrap_for_ema(accelerator, model)

    # Load weights + EMA (in-place copy)
    filename_base = args.filename_base
    model_path = args.model_path or os.path.join(args.checkpoint_dir, f"{filename_base}_model.pt")
    ema_path = args.ema_path or os.path.join(args.checkpoint_dir, f"{filename_base}_ema.pt")

    assert os.path.exists(model_path), f"Cannot find model weights: {model_path}"
    accelerator.print(f"[Load] model: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    unwrapped.load_state_dict(state, strict=True)

    if args.use_ema:
        assert os.path.exists(ema_path), f"Cannot find EMA weights: {ema_path}"
        accelerator.print(f"[Load] ema:   {ema_path}")
        ema_state_cpu = torch.load(ema_path, map_location="cpu")  # {name: tensor}
        missing, unexpected = _apply_ema_inplace_from_state_dict(unwrapped, ema_state_cpu)
        if len(missing):
            accelerator.print(
                f"[EMA] Missing {len(missing)} items, keeping original model params (examples): {missing[:3]}..."
            )
        if len(unexpected):
            accelerator.print(
                f"[EMA] Extra {len(unexpected)} items in ema not used (examples): {unexpected[:3]}..."
            )

    model.eval()

    # Build VAE decode (optional compile)
    decode = build_vae_decode(args, device)

    # Preset: fixed latents and class-balanced labels
    requested_total = int(args.num_images)
    if requested_total % args.num_classes != 0:
        effective_total_imgs = (requested_total // args.num_classes) * args.num_classes
        accelerator.print(
            f"[preset] Adjusting num-images from {requested_total} to {effective_total_imgs} "
            f"(divisible by num-classes={args.num_classes})"
        )
    else:
        effective_total_imgs = requested_total

    y_world_np = np.tile(np.arange(args.num_classes), effective_total_imgs // args.num_classes)
    assert y_world_np.size == effective_total_imgs
    y_world = torch.from_numpy(y_world_np).long().to(device)

    g = torch.Generator(device=device)
    g.manual_seed((args.global_seed if args.global_seed is not None else 0) + int(args.eval_latent_seed))
    x0_world = torch.randn(
        effective_total_imgs, vae_embed_dim, latent_size, latent_size, generator=g, device=device, dtype=torch.float32
    )

    if accelerator.is_main_process:
        bytes_per_sample = vae_embed_dim * latent_size * latent_size * 4
        total_mb = bytes_per_sample * effective_total_imgs / (1024 ** 2)
        accelerator.print(f"[preset] x0_world: shape={tuple(x0_world.shape)}, ~{total_mb:.1f} MB / GPU")

    # Parse interval pairs (global args)
    interval_pairs: Optional[List[Tuple[float, float]]] = None
    if args.cfg_intervals:
        if len(args.cfg_intervals) % 2 != 0:
            raise ValueError("--cfg-intervals must contain an even number of floats (lo hi ...)")
        vals = [float(v) for v in args.cfg_intervals]
        interval_pairs = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
        for lo, hi in interval_pairs:
            assert 0.0 <= lo < hi <= 1.0, f"Interval lo<hi must be within [0,1], got ({lo},{hi})"

    # Combination parameters: schedules × CFG × NFE
    schedule_names = [s.lower() for s in _dedupe_keep_order(args.cfg_schedules)]
    nfe_list = [int(n) for n in _dedupe_keep_order(args.nfe_list)]
    cfg_scales = [float(s) for s in _dedupe_keep_order(args.cfg_scales)]

    assert len(schedule_names) > 0, "--cfg-schedules cannot be empty"
    assert len(nfe_list) > 0, "--nfe-list cannot be empty"
    assert len(cfg_scales) > 0, "--cfg-scales cannot be empty"

    accelerator.print(f"[plan] sampler=euler, schedules={schedule_names}, nfe_list={nfe_list}, cfg_scales={cfg_scales}")
    if "interval" in schedule_names:
        accelerator.print(f"[plan] interval_pairs={interval_pairs or []}")

    eval_batch_size = args.eval_batch_size
    world_bs = eval_batch_size * world
    nsteps_outer = math.ceil(effective_total_imgs / world_bs)

    # Iterate over combinations: sampling -> saving -> evaluation
    for schedule_name in schedule_names:
        sched_tag = _sched_tag(schedule_name, interval_pairs)

        for cfg_scale in cfg_scales:
            # Decide whether to use CFG path (dual-branch) or plain conditional
            want_cfg = (schedule_name in ("linear", "interval")) or (cfg_scale is not None and cfg_scale > 1.0)

            unwrapped = _unwrap_for_ema(accelerator, model)
            has_cfg_impl = hasattr(unwrapped, "forward_with_cfg")
            model_fn = unwrapped.forward_with_cfg if (want_cfg and has_cfg_impl) else unwrapped.forward

            # Build RF + sampler
            rf = RectifiedFlow(
                data_shape=(vae_embed_dim, latent_size, latent_size),
                velocity_field=model_fn,
                device=device,
                dtype=torch.float32,
            )

            for nfe in nfe_list:
                cfg_str = _format_cfg(cfg_scale)

                # filenames (include sampler label to distinguish velocity vs target)
                config_name_full = (
                    f"sampler-euler_nfe-{int(nfe)}_{sched_tag}_cfg-{cfg_str}"
                    f"_n-{effective_total_imgs}"
                    f"_seed-{args.eval_latent_seed}_{'ema' if args.use_ema else 'raw'}"
                )
                config_dir = os.path.join(args.output_dir, config_name_full)
                images_dir = os.path.join(config_dir, "images")

                metrics_short = (
                    f"euler_nfe-{int(nfe)}_{sched_tag}_cfg-{cfg_str}"
                    f"_n-{effective_total_imgs}"
                    f"_seed-{args.eval_latent_seed}_{'ema' if args.use_ema else 'raw'}"
                )
                metrics_json_path = os.path.join(args.output_dir, f"{metrics_short}.json")

                if accelerator.is_main_process:
                    os.makedirs(images_dir, exist_ok=True)
                accelerator.wait_for_everyone()

                sampler = EulerSampler(rf, num_steps=int(nfe))

                accelerator.print(
                    f"[run] sampler=euler, schedule={schedule_name}, nfe={nfe}, cfg={cfg_scale:g}, "
                    f"{('with CFG' if (want_cfg and has_cfg_impl) else 'no CFG')}"
                )
                if schedule_name == "interval":
                    accelerator.print(f"[run] intervals={interval_pairs or []}")
                accelerator.print(f"[run] images -> {images_dir}")
                accelerator.print(f"[run] metrics(JSON) -> {metrics_json_path}")

                # Fixed outer-loop over shards
                t0 = time.time()
                pbar = tqdm(
                    range(nsteps_outer),
                    disable=not accelerator.is_local_main_process,
                    dynamic_ncols=True,
                    desc=f"Sampling {sched_tag} nfe={nfe} cfg={cfg_str}",
                )

                for i in range(nsteps_outer):
                    start = world_bs * i + global_rank * eval_batch_size
                    end = min(start + eval_batch_size, effective_total_imgs)
                    if start >= effective_total_imgs:
                        break

                    y_local = y_world[start:end].clone()
                    z = x0_world[start:end].clone().to(dtype=torch.float32)

                    with torch.inference_mode(), accelerator.autocast():
                        if want_cfg and has_cfg_impl:
                            # Dual-branch (cond/uncond)
                            z_in = torch.cat([z, z], dim=0)
                            y_null = torch.full_like(y_local, args.num_classes)
                            y_infer = torch.cat([y_local, y_null], dim=0)

                            fw_kwargs: Dict[str, Any] = dict(
                                cfg=float(cfg_scale),
                                schedule=schedule_name,
                            )
                            if schedule_name == "interval":
                                fw_kwargs.update(cfg_interval=interval_pairs)

                            latents = sampler.sample_loop(
                                x_0=z_in.clone(),
                                y=y_infer,
                                **fw_kwargs,
                            ).trajectories[-1]
                            latents, _ = latents.chunk(2, dim=0)
                        else:
                            latents = sampler.sample_loop(x_0=z.clone(), y=y_local).trajectories[-1]

                        imgs = decode(latents)

                    imgs = (imgs / 2 + 0.5).clamp_(0.0, 1.0).detach().cpu()
                    for b in range(imgs.size(0)):
                        gid = i * world_bs + global_rank * eval_batch_size + b
                        if gid >= effective_total_imgs:
                            break
                        save_image(imgs[b], os.path.join(images_dir, f"{gid:06d}.png"))

                    del imgs, latents, z, y_local
                    pbar.update(1)

                accelerator.wait_for_everyone()
                t1 = time.time()

                # ---------------------------
                # Two-pass metrics in *samples root*
                # ---------------------------
                if accelerator.is_main_process:
                    result = OrderedDict(
                        sampler="euler",
                        cfg_schedule=schedule_name,
                        cfg_scale=float(cfg_scale),
                        cfg_intervals=(interval_pairs if schedule_name == "interval" else None),
                        nfe=int(nfe),
                        num_images=int(effective_total_imgs),
                        seconds=float(t1 - t0),
                        fid=None,
                        inception_score_mean=None,
                        inception_score_std=None,
                        precision=None,
                        recall=None,
                        fid_stats_path=None,
                        real_images_path=None,
                        images_dir=images_dir,
                        images_parent=config_dir,
                        metrics_json=metrics_json_path,
                    )

                    # Pass 1: FID & IS
                    fid_ok = os.path.exists(args.fid_stats_path) if args.fid_stats_path else False
                    try:
                        metrics_main = torch_fidelity.calculate_metrics(
                            input1=images_dir,
                            input2=None,
                            fid_statistics_file=(args.fid_stats_path if fid_ok else None),
                            cuda=True,
                            isc=True,
                            fid=fid_ok,
                            kid=False,
                            prc=False,
                            verbose=False,
                        )
                        result["fid"] = float(metrics_main.get("frechet_inception_distance")) if fid_ok else None
                        result["inception_score_mean"] = float(metrics_main.get("inception_score_mean"))
                        result["inception_score_std"] = float(metrics_main.get("inception_score_std"))
                        result["fid_stats_path"] = args.fid_stats_path if fid_ok else None
                    except Exception as e:
                        accelerator.print(f"[metrics:main FID/IS] Calculation failed: {e}")

                    # Pass 2: Precision/Recall (requires real 10k folder)
                    prc_ok = bool(args.real_images_path) and os.path.isdir(args.real_images_path)
                    if prc_ok:
                        try:
                            metrics_prc = torch_fidelity.calculate_metrics(
                                input1=args.real_images_path,
                                input2=images_dir,
                                fid=False,
                                isc=False,
                                kid=False,
                                prc=True,
                                cuda=True,
                                verbose=False,
                            )
                            result["precision"] = float(metrics_prc.get("precision"))
                            result["recall"] = float(metrics_prc.get("recall"))
                            result["real_images_path"] = args.real_images_path
                        except Exception as e:
                            accelerator.print(f"[metrics:PRC] Calculation failed: {e}")
                    else:
                        accelerator.print("[metrics:PRC] Skipped (set --real-images-path to 10k reference folder)")

                    # JSON in samples root (only)
                    with open(metrics_json_path, "w", encoding="utf-8") as jf:
                        json.dump(result, jf, ensure_ascii=False, indent=2)

                    accelerator.print(
                        f"[done] {metrics_short} | FID={result['fid']} "
                        f"| IS={result['inception_score_mean']}±{result['inception_score_std']} "
                        f"| PRC=({result['precision']},{result['recall']}) "
                        f"| {result['seconds']:.1f}s"
                    )

                    # Optional: cleanup images to save storage
                    if args.cleanup_images:
                        try:
                            shutil.rmtree(images_dir)
                            accelerator.print(f"[cleanup] Deleted images dir: {images_dir}")
                        except Exception as e:
                            accelerator.print(f"[cleanup] Failed to delete {images_dir}: {e}")

                accelerator.wait_for_everyone()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    accelerator.print("All sampling finished.")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
