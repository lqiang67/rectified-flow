# LightningDiT training with Hugging Face Accelerate + bf16 + torch.compile
# ---------------------------------------------------------------
# - EMA kept as lightweight tensors (param_name list + tensor list), not a Module.
# - We only swap to EMA weights **in-place on GPU at sampling time** (online eval).
# - accelerate.save_state() handles optimizer/schedulers; we save model + EMA tensors
#   as separate pt files: {filename_base}_model.pt and {filename_base}_ema.pt.
# - torch.compile is applied to the training model only.
# - Rectified-Flow training unchanged; online eval uses EMA via in-place swap.

import os
import gc
import math
import time
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from rectified_flow.models.lightningdit import LightningDiT_models
from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers.euler_sampler import EulerSampler

from diffusers.models import AutoencoderKL as SDAutoencoderKL
try:
    from rectified_flow.models.vae import AutoencoderKL as MARAutoencoderKL, DiagonalGaussianDistribution
    from rectified_flow.models.vae import CachedFolder
except Exception:
    MARAutoencoderKL = None
    DiagonalGaussianDistribution = None
    CachedFolder = None

from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
)

import torch_fidelity
import shutil
from tqdm.auto import tqdm, trange


# ---------------------------
# EMA helpers
# ---------------------------
@torch.no_grad()
def _load_ema_on_gpu_inplace(unwrapped, param_names, ema_params):
    """
    Copy EMA tensors into the model's parameters in-place (stays on GPU).
    Returns a GPU snapshot (list of cloned parameter tensors) for restoration.
    """
    named = dict(unwrapped.named_parameters())
    snapshot = []
    for name, ema_t in zip(param_names, ema_params):
        p = named[name]
        snapshot.append(p.data.detach().clone())  # same device/dtype as p
        p.data.copy_(ema_t.to(device=p.device, dtype=p.dtype, non_blocking=True))
    return snapshot

@torch.no_grad()
def _restore_from_gpu_snapshot(unwrapped, param_names, snapshot):
    named = dict(unwrapped.named_parameters())
    for name, snap in zip(param_names, snapshot):
        named[name].data.copy_(snap, non_blocking=True)

@torch.no_grad()
def update_ema_params(ema_params, src_params, decay=0.9999):
    """
    Tensor-only EMA update: ema <- decay * ema + (1-decay) * src
    `src_params` must be references to the model's parameter `.data`.
    """
    for ema_t, src_t in zip(ema_params, src_params):
        ema_t.mul_(decay).add_(src_t, alpha=1.0 - decay)

@torch.no_grad()
def init_ema_from_model(unwrapped):
    """
    Build (param_names, src_params, ema_params) from an unwrapped (possibly compiled) model.
    - param_names: list[str], in the same order as named_parameters()
    - src_params:  list[Tensor], references to p.data for in-place EMA updates
    - ema_params:  list[Tensor], cloned tensors initialized from model weights
    """
    param_names, src_params, ema_params = [], [], []
    for n, p in unwrapped.named_parameters():
        param_names.append(n)
        src_params.append(p.data)
        ema_params.append(p.data.detach().clone())
    return param_names, src_params, ema_params


# ---------------------------
# LR schedule helpers (epoch-based warmup)
# ---------------------------
def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup (epoch-based).
    Warmup: linear from 0 -> args.lr over args.warmup_epochs.
    After warmup:
        - 'constant': stay at args.lr
        - 'cosine'  : cosine from args.lr -> args.min_lr over (args.epochs - args.warmup_epochs)
    """
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        lr = args.lr * (epoch / float(args.warmup_epochs))
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            total_after = max(1, args.epochs - max(0, args.warmup_epochs))
            progress = min(max(0, epoch - max(0, args.warmup_epochs)), total_after)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * progress / total_after))
        else:
            raise NotImplementedError(f"Unknown lr_schedule: {args.lr_schedule}")
    for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr


# ---------------------------
# Weight decay param-groups (no decay on LayerNorm & bias)
# ---------------------------
def add_weight_decay(model, weight_decay: float = 0.01, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mod_name, _, pname = name.rpartition(".")
        mod = model.get_submodule(mod_name) if mod_name else model

        is_bias = (pname == "bias") or name.endswith(".bias")
        is_layernorm = isinstance(mod, nn.LayerNorm)

        if is_bias or is_layernorm or (name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay,    "weight_decay": weight_decay},
    ]


def center_crop_arr(pil_image, image_size):
    """
    Center crop from ADM reference implementation.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# -------------
# Online evaluation (FID, IS, Precision, Recall) — swap to EMA weights during sampling
# Uses pre-sampled, device-resident x0_world (latents) and y_world (labels).
# -------------
@torch.no_grad()
def online_eval_ldit(
    args,
    accelerator,
    model_wrapped,              # accelerate-wrapped (possibly compiled) model
    param_names, ema_params,    # EMA tensor state
    decode,
    device,
    eval_batch_size,
    world,
    global_rank,
    global_step,
    cfg_value,
    latent_size,
    vae_embed_dim,
    x0_world,                   # [N, C, H, W] on device
    y_world,                    # [N] on device (Long)
    tag: str = ""
):
    if not args.online_eval or args.eval_every <= 0:
        return

    # Total pre-sampled images
    total_imgs = int(y_world.numel())
    assert x0_world.shape[0] == total_imgs, "x0_world/y_world length mismatch"
    assert x0_world.shape[1:] == (vae_embed_dim, latent_size, latent_size), "x0_world shape mismatch"

    per_step_world = eval_batch_size * world
    if per_step_world == 0:
        accelerator.print("[online-eval] per-step world batch is 0; check eval-batch-size/global-batch-size.")
        return
    num_steps = math.ceil(total_imgs / per_step_world)

    save_root = os.path.join(args.output_dir, "eval")
    if accelerator.is_main_process:
        os.makedirs(save_root, exist_ok=True)
    accelerator.wait_for_everyone()

    cfg_str = "1.0" if (cfg_value is None) else f"{cfg_value:g}"
    save_dir = os.path.join(
        save_root,
        f"step{global_step:07d}-cfg{cfg_str}-n{total_imgs}_ema{('-' + tag) if tag else ''}"
    )
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    def _unwrap(m):
        u = accelerator.unwrap_model(m)
        return u._orig_mod if is_compiled_module(u) else u

    # Unwrap and swap to EMA in-place on GPU
    unwrapped = _unwrap(model_wrapped)
    was_training = unwrapped.training
    snapshot_gpu = _load_ema_on_gpu_inplace(unwrapped, param_names, ema_params)

    try:
        unwrapped.eval()
        use_cfg = (cfg_value is not None) and (cfg_value > 1.0)
        model_fn = unwrapped.forward_with_cfg if use_cfg and hasattr(unwrapped, "forward_with_cfg") else unwrapped.forward

        rf_sample = RectifiedFlow(
            data_shape=(vae_embed_dim, latent_size, latent_size),
            velocity_field=model_fn,
            device=device,
            dtype=torch.float32,
        )
        sampler = EulerSampler(rf_sample, num_steps=getattr(args, "eval_num_steps", 100))

        for i in trange(
            num_steps, desc="Eval sampling",
            disable=not accelerator.is_local_main_process, dynamic_ncols=True
        ):
            start = per_step_world * i + global_rank * eval_batch_size
            end   = min(start + eval_batch_size, total_imgs)
            if start >= total_imgs:
                break

            # Fixed per-rank slice (same across runs)
            y_local = y_world[start:end].clone()                     # Long, on device
            z = x0_world[start:end].clone().to(dtype=torch.float32)  # Float32, on device

            with torch.inference_mode(), accelerator.autocast():
                if use_cfg and hasattr(unwrapped, "forward_with_cfg"):
                    z_in = torch.cat([z, z], dim=0)  # duplicate latents for cond/uncond
                    y_null = torch.full_like(y_local, args.num_classes)
                    y_infer = torch.cat([y_local, y_null], dim=0)
                    latents = sampler.sample_loop(x_0=z_in.clone(), y=y_infer, cfg=cfg_value).trajectories[-1]
                    latents, _ = latents.chunk(2, dim=0)
                else:
                    latents = sampler.sample_loop(x_0=z.clone(), y=y_local).trajectories[-1]

                imgs = decode(latents)

            imgs = (imgs / 2 + 0.5).clamp_(0.0, 1.0).detach().cpu()
            for b in range(imgs.size(0)):
                gid = i * per_step_world + global_rank * eval_batch_size + b
                if gid >= total_imgs:
                    break
                save_image(imgs[b], os.path.join(save_dir, f"{gid:06d}.png"))

            del imgs, latents, z, y_local

        accelerator.wait_for_everyone()

        # Rank-0 computes metrics (two-pass: FID/IS, then PRC)
        if accelerator.is_main_process:
            suffix = "_ema" + ("" if (cfg_value is None or cfg_value == 1.0) else f"_cfg{cfg_value:g}")
            log_payload, msg = {}, "[online-eval]"

            # -------- Pass 1: FID (if stats available) + IS (always) --------
            fid_ok = os.path.exists(args.fid_stats_path) if getattr(args, "fid_stats_path", None) else False
            fid = None
            is_mean = None
            try:
                metrics_main = torch_fidelity.calculate_metrics(
                    input1=save_dir,
                    input2=None,
                    fid_statistics_file=(args.fid_stats_path if fid_ok else None),
                    cuda=True,
                    isc=True,
                    fid=fid_ok,           
                    kid=False,
                    prc=False,
                    verbose=False,
                )
                if fid_ok:
                    fid = metrics_main.get("frechet_inception_distance", None)
                is_mean = metrics_main.get("inception_score_mean", None)
                del metrics_main
            except Exception as e:
                accelerator.print(f"[online-eval] FID/IS calculation failed: {e}")

            if fid is not None:
                log_payload[f"eval/fid{suffix}"] = float(fid)
                msg += f" FID{suffix}={fid:.4f}"
            else:
                if not fid_ok:
                    accelerator.print(f"[online-eval] FID stats not found at '{args.fid_stats_path}'. FID skipped.")
            if is_mean is not None:
                log_payload[f"eval/is{suffix}"] = float(is_mean)
                msg += f" IS{suffix}={is_mean:.4f}"

            # -------- Pass 2: Precision / Recall (需要 real_images_path 目录) --------
            precision = None
            recall = None
            prc_ok = bool(getattr(args, "real_images_path", "")) and os.path.isdir(args.real_images_path)
            if prc_ok:
                try:
                    metrics_prc = torch_fidelity.calculate_metrics(
                        input1=args.real_images_path,   # Reference 10k real images
                        input2=save_dir,                # Generated images folder
                        fid=False,
                        isc=False,
                        kid=False,
                        prc=True,
                        cuda=True,
                        verbose=False,
                    )
                    precision = metrics_prc.get("precision", None)
                    recall = metrics_prc.get("recall", None)
                    del metrics_prc
                except Exception as e:
                    accelerator.print(f"[online-eval] PRC calculation failed: {e}")
            else:
                accelerator.print(f"[online-eval] real_images_path '{getattr(args, 'real_images_path', None)}' not found. Skip Precision/Recall.")

            if precision is not None:
                log_payload[f"eval/precision{suffix}"] = float(precision)
                msg += f" Precision{suffix}={precision:.4f}"
            if recall is not None:
                log_payload[f"eval/recall{suffix}"] = float(recall)
                msg += f" Recall{suffix}={recall:.4f}"

            accelerator.print(msg)

            if args.report_to != "none" and len(log_payload):
                accelerator.log(log_payload, step=global_step)

            # 可选清理生成图
            if not args.keep_eval_images:
                shutil.rmtree(save_dir, ignore_errors=True)

        accelerator.wait_for_everyone()
    finally:
        # Restore original training weights and mode
        _restore_from_gpu_snapshot(unwrapped, param_names, snapshot_gpu)
        unwrapped.train(was_training)
        del snapshot_gpu
        gc.collect()
        torch.cuda.empty_cache()


# -------------
# Main training
# -------------
def main(args):
    # ---------------------------
    # Accelerator configuration
    # ---------------------------
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    logging_dir = os.path.join(args.output_dir, args.logging_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    dynamo = None
    if args.torch_compile:
        from accelerate.utils import TorchDynamoPlugin
        dynamo = TorchDynamoPlugin(
            backend=args.compile_backend,
            mode=args.compile_mode,
            fullgraph=False,
        )
        print(f"[TorchDynamo] Enabled with backend={args.compile_backend}, mode={args.compile_mode}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,          # 'bf16' by default
        dynamo_plugin=dynamo,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    device = accelerator.device
    set_seed(args.global_seed, device_specific=True)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    # Batch sizes
    world = accelerator.num_processes
    denom = world * args.gradient_accumulation_steps
    assert args.global_batch_size % denom == 0, \
        f"--global-batch-size ({args.global_batch_size}) must be divisible by world_size({world}) * grad_accum_steps({args.gradient_accumulation_steps})."
    local_batch_size = args.global_batch_size // denom

    world_size  = accelerator.num_processes
    global_rank = accelerator.process_index
    local_rank  = accelerator.local_process_index
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE")
        or os.environ.get("SLURM_GPUS_ON_NODE","1").split('(')[0].split(',')[0]
        or torch.cuda.device_count()
    )
    num_nodes = max(1, world_size // max(1, local_world_size))
    accelerator.print(
        f"[dist] rank={global_rank}, "
        f"local_rank={local_rank}, "
        f"world_size={world_size}, "
        f"num_nodes={num_nodes}, "
        f"gpus_per_node={local_world_size}"
    )

    # ---------------------------
    # Model / VAE / sizes
    # ---------------------------
    if args.vae_impl == "sd":
        vae_stride = args.vae_stride or 8
        vae_embed_dim = args.vae_embed_dim or 4
    else:  # "mar"
        vae_stride = args.vae_stride or 16
        vae_embed_dim = args.vae_embed_dim or 16

    assert args.image_size % vae_stride == 0, f"Image size must be divisible by {vae_stride}"
    latent_size = args.image_size // vae_stride

    class_drop = args.class_dropout_prob
    if class_drop is None:
        class_drop = 0.1 if args.cfg_scale > 1.0 else 0.0

    base_model = LightningDiT_models[args.model](
        input_size=latent_size,
        patch_size=args.patch_size,
        in_channels=vae_embed_dim,  # VAE latent: 4 or 16 × (H/stride) × (W/stride)
        num_classes=args.num_classes,
        class_dropout_prob=class_drop,
        time_scale=args.time_scale,
        use_qknorm=True,
    )

    # --- optimizer with param groups (exclude LayerNorm & bias from WD) ---
    param_groups = add_weight_decay(base_model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        lr=args.lr,
        fused=args.use_fused_adam,
        eps=args.adam_epsilon,
    )

    # ---------------------------
    # Data pipeline (ImageNet-256)
    # ---------------------------
    if args.use_cached:
        assert args.vae_impl == "mar", "--use_cached only supports --vae-impl=mar"
        assert CachedFolder is not None, "CachedFolder not found"
        assert args.cached_path and os.path.isdir(args.cached_path), "Please provide a valid --cached_path"
        dataset = CachedFolder(args.cached_path)  # (moments, label)
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil: center_crop_arr(pil, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        dataset = ImageFolder(args.data_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,                 # Accelerate will replace with DistributedSampler automatically
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    # ---------------------------
    # Prepare (Accelerate wraps / compiles)
    # ---------------------------
    model, optimizer, dataloader = accelerator.prepare(base_model, optimizer, dataloader)

    # Helper to unwrap compiled/accelerated module
    def _unwrap_for_save_or_ema(m):
        m = accelerator.unwrap_model(m)
        return m._orig_mod if is_compiled_module(m) else m

    # Build EMA storage after prepare (to match wrapped params)
    unwrapped_for_ema = _unwrap_for_save_or_ema(model)
    param_names, src_params, ema_params = init_ema_from_model(unwrapped_for_ema)  # step-0 EMA = model weights

    # ---------------------------
    # VAE and compile encode/decode functions
    # ---------------------------
    if args.vae_impl == "sd":
        # stabilityai/sd-vae-ft-<mse|ema>, scale 0.18215
        vae = SDAutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
        VAE_SCALE = 0.18215
        vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if args.compile_vae:
            def _enc(x): return vae.encode(x.float()).latent_dist.sample() * VAE_SCALE
            def _dec(z): return vae.decode(z.float() / VAE_SCALE).sample
            encode = torch.compile(_enc, backend=args.compile_backend, mode=args.compile_mode)
            decode = torch.compile(_dec, backend=args.compile_backend, mode=args.compile_mode)
            accelerator.print("[compile] SD VAE encode/decode compiled")
        else:
            def encode(x): return vae.encode(x.float()).latent_dist.sample() * VAE_SCALE
            def decode(z): return vae.decode(z.float() / VAE_SCALE).sample
    else:
        # MAR's VAE, scale 0.2325
        assert MARAutoencoderKL is not None, "Could not find models.vae.AutoencoderKL (MAR)"
        vae = MARAutoencoderKL(embed_dim=vae_embed_dim, ch_mult=(1, 1, 2, 2, 4),
                               ckpt_path=args.vae_path).to(device).eval()
        for p in vae.parameters(): p.requires_grad = False
        VAE_SCALE = 0.2325
        if args.compile_vae:
            def _enc(x): return vae.encode(x.float()).sample() * VAE_SCALE
            def _dec(z): return vae.decode(z.float() / VAE_SCALE)
            encode = torch.compile(_enc, backend=args.compile_backend, mode=args.compile_mode)
            decode = torch.compile(_dec, backend=args.compile_backend, mode=args.compile_mode)
            accelerator.print("[compile] MAR VAE encode/decode compiled")
        else:
            def encode(x): return vae.encode(x.float()).sample() * VAE_SCALE
            def decode(z): return vae.decode(z.float() / VAE_SCALE)

    # Rectified Flow utilities for training
    rf_train = RectifiedFlow(
        data_shape=(vae_embed_dim, latent_size, latent_size),
        velocity_field=model,      # we call the wrapped model directly in training
        device=device,
        dtype=torch.float32,
        train_time_distribution=args.train_time_distribution,
    )

    # ---------------------------
    # Trackers (W&B / TensorBoard)
    # ---------------------------
    if accelerator.is_main_process and args.report_to != "none":
        run_name = f"{args.model}-bf16-qkvnorm-{timestamp}-ema-state-only"
        accelerator.init_trackers(
            project_name=args.tracker_project,
            config=vars(args),
            init_kwargs={"wandb": {
                "name": run_name,
                "resume": "allow",
                "dir": logging_dir,
            }}
        )

    accelerator.print(f"LDiT Parameters: {sum(p.numel() for p in accelerator.unwrap_model(model).parameters()):,}")
    accelerator.print(f"ImageNet Dataset size: {len(dataset):,} images at {args.data_path}")

    # ---------------
    # Save/Load Hooks (save model state + EMA tensor map)
    # ---------------
    filename_base = "ldit"

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unwrapped = _unwrap_for_save_or_ema(models[0])
            merged_state_dict = accelerator.get_state_dict(unwrapped)  # CPU-safe under FSDP/Zero3
            model_path = os.path.join(output_dir, f"{filename_base}_model.pt")
            ema_path = os.path.join(output_dir, f"{filename_base}_ema.pt")

            # Save training model
            torch.save(merged_state_dict, model_path)

            # Save EMA as {name: tensor(cpu)}
            ema_state_cpu = OrderedDict(
                (name, ema_params[i].detach().to("cpu")) for i, name in enumerate(param_names)
            )
            torch.save(ema_state_cpu, ema_path)
            accelerator.print(f"[Checkpoint] Saved model & EMA tensors to {output_dir}")
        weights.clear()  # let Accelerate save optimizer/schedulers separately

    def load_model_hook(models, input_dir):
        nonlocal ema_params, param_names, src_params
        unwrapped = _unwrap_for_save_or_ema(models[0])
        model_path = os.path.join(input_dir, f"{filename_base}_model.pt")
        ema_path = os.path.join(input_dir, f"{filename_base}_ema.pt")
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            unwrapped.load_state_dict(state, strict=True)

            # Refresh mappings (in case shapes/order changed)
            param_names = [n for n, _ in unwrapped.named_parameters()]
            src_params = [p.data for _, p in unwrapped.named_parameters()]

            if os.path.exists(ema_path):
                ema_state_cpu = torch.load(ema_path, map_location="cpu")  # {name: tensor}
                new_ema = []
                for name, src in zip(param_names, src_params):
                    if name in ema_state_cpu:
                        new_ema.append(ema_state_cpu[name].to(device=src.device, dtype=src.dtype).detach().clone())
                    else:
                        new_ema.append(src.detach().clone())
                        accelerator.print(f"[Checkpoint] EMA param '{name}' not found in checkpoint; init from model weights.")
                ema_params = new_ema
            else:
                ema_params = [s.detach().clone() for s in src_params]

            accelerator.print(f"[Checkpoint] Loaded model and EMA (if present) from {input_dir}")
            while len(models) > 0:
                models.pop()
        else:
            accelerator.print(f"[Checkpoint] No '{filename_base}_model.pt' in {input_dir}; skip custom load.")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ----------------
    # Resume training?
    # ----------------
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            resume_dir = args.resume_from_checkpoint.rstrip("/")
            if not os.path.isdir(resume_dir):
                resume_dir = None
        else:
            candidates = [os.path.join(args.output_dir, d)
                          for d in os.listdir(args.output_dir)
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))]
            resume_dir = max(candidates, key=os.path.getmtime) if candidates else None

        if resume_dir is None:
            accelerator.print(f"[Resume] '{args.resume_from_checkpoint}' not found. Start fresh.")
        else:
            accelerator.print(f"[Resume] Loading state from {resume_dir}")
            accelerator.load_state(resume_dir)
            base = os.path.basename(resume_dir)
            try:
                global_step = int(base.split("-")[1])
            except Exception:
                global_step = 0

    # -----------------------------
    # Training / Logging parameters
    # -----------------------------
    model.train()   # enable dropout for CFG, etc.

    running_loss = 0.0
    log_steps = 0
    last_log_time = time.time()

    eval_batch_size = min(local_batch_size, args.eval_batch_size or local_batch_size)

    # -----------------------------------------
    # Pre-sample & fix eval x0_world / y_world
    # -----------------------------------------
    # Round total eval images to multiple of num_classes for class-balanced labels.
    requested_total = int(args.eval_num_images)
    if requested_total % args.num_classes != 0:
        effective_total_imgs = (requested_total // args.num_classes) * args.num_classes
        accelerator.print(f"[online-eval] Adjust eval-num-images from {requested_total} -> {effective_total_imgs} (divisible by num-classes={args.num_classes}).")
    else:
        effective_total_imgs = requested_total

    # Fixed, class-balanced labels across all ranks (replicated per GPU)
    y_world_np = np.tile(np.arange(args.num_classes), effective_total_imgs // args.num_classes)
    assert y_world_np.size == effective_total_imgs
    y_world = torch.from_numpy(y_world_np).long().to(device)

    # Fixed latents across all ranks (replicated per GPU, on device)
    # Use an explicit seed to ensure the tensor values are identical on each rank and consistent across runs.
    g = torch.Generator(device=device)
    g.manual_seed((args.global_seed if args.global_seed is not None else 0) + int(args.eval_latent_seed))
    x0_world = torch.randn(
        effective_total_imgs, vae_embed_dim, latent_size, latent_size,
        generator=g, device=device, dtype=torch.float32
    )

    # For info/debug
    if accelerator.is_main_process:
        bytes_per_sample = vae_embed_dim * latent_size * latent_size * 4  # float32
        total_mb = bytes_per_sample * effective_total_imgs / (1024**2)
        accelerator.print(f"[online-eval] Fixed x0_world on device: shape={tuple(x0_world.shape)}, ~{total_mb:.1f} MB per GPU")

    # ----------------
    # Training loop
    # ----------------
    total_updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * total_updates_per_epoch

    resume_micro_step = 0
    if global_step > 0:
        first_epoch = global_step // total_updates_per_epoch
        updates_done_in_epoch = global_step % total_updates_per_epoch
        resume_micro_step = updates_done_in_epoch * args.gradient_accumulation_steps
        accelerator.print(
            f"[Resume] first_epoch={first_epoch}, "
            f"updates_done_in_epoch={updates_done_in_epoch}, "
            f"skip_micro_steps={resume_micro_step}"
        )
    else:
        first_epoch = 0

    pbar = tqdm(
        range(global_step, max_train_steps),
        initial=global_step,
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="Train steps",
        dynamic_ncols=True,
    )

    ema_loss = None
    ema_beta = 0.98

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(first_epoch, args.epochs):
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
            
        for step, batch in enumerate(dataloader):
            if epoch == first_epoch and global_step > 0 and step < resume_micro_step:
                continue

            progress = epoch + (step + 1) / len(dataloader)
            cur_lr = adjust_learning_rate(optimizer, progress, args)

            with accelerator.accumulate(model):
                x, y = batch
                # VAE encode images to latents
                with torch.no_grad():
                    if args.use_cached:
                        posterior = DiagonalGaussianDistribution(x)
                        x_1 = posterior.sample().mul_(VAE_SCALE)
                    else:
                        x_1 = encode(x.float())

                x_0 = torch.randn_like(x_1)
                t = rf_train.sample_train_time(x_0.size(0))
                x_t, dot_x_t = rf_train.get_interpolation(x_0=x_0, x_1=x_1, t=t)

                with accelerator.autocast():
                    v_pred = model(x_t, t, y)

                loss = torch.nn.functional.mse_loss(v_pred.float(), dot_x_t.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item())
            log_steps += 1

            if accelerator.sync_gradients:
                update_ema_params(ema_params, src_params, decay=args.ema_decay_rate)

                loss_val = float(loss.detach().item())
                ema_loss = loss_val if ema_loss is None else ema_beta * ema_loss + (1.0 - ema_beta) * loss_val
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "ema_loss": f"{ema_loss:.4f}",
                    "lr": f"{cur_lr:.2e}",
                })
                if accelerator.is_main_process and args.report_to != "none":
                    accelerator.log({"train/lr": float(optimizer.param_groups[0]["lr"]),
                                     "train/epoch": epoch}, step=global_step)
                    
                # periodic checkpoint
                if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        os.makedirs(ckpt_dir, exist_ok=True)
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        accelerator.print(f"[Checkpoint] Saved accelerate state to {ckpt_dir}")

                # periodic online eval FID/IS/PRC — swap to EMA, sample, then restore
                if args.online_eval and args.eval_every > 0 and global_step % args.eval_every == 0:
                    online_eval_ldit(
                        args=args,
                        accelerator=accelerator,
                        model_wrapped=model,
                        param_names=param_names,
                        ema_params=ema_params,
                        decode=decode,
                        device=device,
                        eval_batch_size=eval_batch_size,
                        world=world,
                        global_rank=global_rank,
                        global_step=global_step,
                        cfg_value=args.cfg_scale,
                        latent_size=latent_size,
                        vae_embed_dim=vae_embed_dim,
                        x0_world=x0_world,
                        y_world=y_world,
                        tag="",
                    )

                # periodic scalar logging
                if args.log_every > 0 and global_step % args.log_every == 0:
                    now = time.time()
                    steps_per_sec = log_steps / max(now - last_log_time, 1e-8)
                    avg_loss_local = running_loss / max(log_steps, 1)
                    avg_loss_tensor = torch.tensor(avg_loss_local, device=device)
                    avg_loss_global = accelerator.gather(avg_loss_tensor).mean().item()

                    accelerator.print(f"(step={global_step:07d}) Train Loss: {avg_loss_global:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                    if args.report_to != "none":
                        accelerator.log(
                            {"train/loss": avg_loss_global, "train/steps_per_sec": steps_per_sec},
                            step=global_step
                        )

                    running_loss = 0.0
                    log_steps = 0
                    last_log_time = now

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    model.eval()
    accelerator.print("Training done.")
    accelerator.end_training()


# -----------
# Arg parser
# -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accelerate-style LDiT training on ImageNet-256 with bf16 + torch.compile (EMA as tensor-only state, fixed eval latents)")

    # Data / model
    parser.add_argument("--data-path", type=str, help="ImageNet-256 download/cache directory", default="./data/imagenet256")
    parser.add_argument("--output-dir", type=str, default="./imagenet256-LDiT-bf16-qkvnorm",)
    parser.add_argument("--model", type=str, choices=LightningDiT_models, default="LightningDiT-XL")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1,
                        help="Class dropout probability for classifier-free guidance. Default 0.1 if --cfg-scale > 1.0 else 0.0")
    parser.add_argument("--patch-size", type=int, default=1)

    # Training schedule
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--global-batch-size", type=int, default=2048, help="Effective global batch per optimizer step (includes world_size * grad_accum).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--use-fused-adam", action="store_true", help="Use fused AdamW from apex or torch (if available).")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=3.0, help="Max gradient norm for clipping. Set to 0 to disable.")
    parser.add_argument("--ema-decay-rate", type=float, default=0.9999, help="EMA decay rate for model weights.")
    # LR schedule (epoch-based warmup)
    parser.add_argument("--warmup-epochs", type=int, default=32, help="Linear warmup epochs (epoch-based).")
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"],
                        help="LR schedule after warmup (default: constant).")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine schedule.")

    # VAE
    parser.add_argument("--vae-impl", type=str, default="mar", choices=["sd", "mar"],
                        help="Choose VAE implementation: sd=stabilityai/sd-vae-ft-*, mar=models.vae.AutoencoderKL (MAR)")
    parser.add_argument("--vae", type=str, default="ema", choices=["mse", "ema"],
                        help="When --vae-impl=sd, load sd-vae-ft-<vae> (default ema).")
    parser.add_argument("--vae-path", type=str, default="./pretrained_models/vae/kl16.ckpt",
                        help="MAR VAE's ckpt path (only used when --vae-impl=mar).")
    parser.add_argument("--compile-vae", action="store_true", default=True)
    parser.add_argument("--vae-embed-dim", type=int, default=16,
                        help="VAE latent channel dimension: sd default=4, mar default=16")
    parser.add_argument("--vae-stride", type=int, default=16,
                        help="VAE downsampling stride: sd default=8, mar default=16")
    parser.add_argument("--use-cached", action="store_true", default=True,
                        help="Use MAR cached latents (only available when --vae-impl=mar).")
    parser.add_argument("--cached-path", type=str, default="./data/imagenet_kl16_latent",
                        help="MAR cache moments root directory (used with --use_cached).")

    # Rectified Flow config
    parser.add_argument("--train-time-distribution", type=str, default="lognormal",
                        choices=["uniform", "u_shaped", "lognormal"], help="Time sampling distribution for training.")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Time scale for SiT time embedding.")

    # Logging / checkpoints / sampling
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25_000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--tracker-project", type=str, default="SiT-accelerate-ImageNet256")
    parser.add_argument("--logging-dir", type=str, default="logs")

    # Online evaluation (FID / IS / PRC)
    parser.add_argument("--online-eval", action="store_true", default=True,
                        help="Enable online FID/IS evaluation with EMA.")
    parser.add_argument("--eval-every", type=int, default=50_000,
                        help="Run online eval every N optimizer steps; 0 disables.")
    parser.add_argument("--eval-num-images", type=int, default=50_000,
                        help="Total #images across ALL processes for FID/IS (rounded to multiples of num-classes).")
    parser.add_argument("--eval-batch-size", type=int, default=100,
                        help="Per-device batch size for evaluation sampling.")
    parser.add_argument("--eval-num-steps", type=int, default=64,
                        help="EulerSampler steps used in online eval sampling.")
    parser.add_argument("--fid-stats-path", type=str, default="./fid_stats/adm_in256_stats.npz",
                        help="Precomputed FID stats for ImageNet-256.")
    parser.add_argument("--keep-eval-images", action="store_true", default=True,
                        help="Keep generated images after metrics computed.")
    parser.add_argument("--eval-latent-seed", type=int, default=2333,
                        help="Seed used to pre-sample fixed eval latents (x0_world).")
    parser.add_argument("--real-images-path", type=str,
                        default="./data/ref_images_imnet256",
                        help="Directory of 10k reference real images for Precision/Recall. If missing, PRC is skipped.")

    # Accelerate / precision / compile / TF32
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true", default=True)
    parser.add_argument("--torch-compile", action="store_true", default=True)
    parser.add_argument("--compile-backend", type=str, default="inductor", choices=["inductor", "torch_tensorrt"])
    parser.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])

    # Resume
    parser.add_argument("--resume-from-checkpoint", type=str, default="latest",
                        help='Path to a checkpoint dir saved by accelerate, or "latest" to auto-pick the newest.')

    args = parser.parse_args()
    main(args)
