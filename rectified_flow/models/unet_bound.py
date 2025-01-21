# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".
Copied from https://github.com/NVlabs/edm/blob/main/training/networks.py
"""

import numpy as np
import torch
import os
import json

from torch.nn.functional import silu
from dataclasses import dataclass, asdict, field
from rectified_flow.utils import match_dim_with_data

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, t, time_factor=1000.0):
        t = t * time_factor
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=t.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        t = t.ger(freqs.to(t.dtype))
        t = torch.cat([t.cos(), t.sin()], dim=1)
        return t


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@dataclass
class SongUNetConfig:
    img_resolution: int     # Image resolution at input/output.
    in_channels: int        # Number of color channels at input.
    out_channels: int       # Number of color channels at output.
    label_dim: int = 0      # Number of class labels, 0 = unconditional.
    augment_dim: int = 0    # Augmentation label dimensionality, 0 = no augmentation.

    model_channels: int = 128             # Base multiplier for the number of channels.
    channel_mult: list[int] = field(default_factory=lambda: [2, 2, 2])   
    channel_mult_emb: int = 4             # Multiplier for the dimensionality of the embedding vector.
    num_blocks: int = 4                   # Number of residual blocks per resolution.
    attn_resolutions: list[int] = field(default_factory=lambda: [16])    
    dropout: float = 0.13                 # Dropout probability of intermediate activations.
    label_dropout: float = 0.0            # Dropout probability of class labels for classifier-free guidance.
    embedding_type: str = "positional"    # Timestep embedding type: 'positional' or 'fourier'.
    channel_mult_time: int = 1            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
    encoder_type: str = "standard"        # Encoder architecture: 'standard' or 'residual'.
    decoder_type: str = "standard"        # Decoder architecture: 'standard' or 'residual'.
    resample_filter: list[int] = field(default_factory=lambda: [1, 1])   

    bound_type: str = "none"              # Boundary condition type: 'none', 'single', or 'double'
    bound_order: float | str = None       # Order of the boundary condition: 1 or 2, or 'sin'


class SongUNet(torch.nn.Module):
    config_class = SongUNetConfig

    def __init__(
        self,
        config: SongUNetConfig,
        data_mean: torch.Tensor = None,
    ):
        self.config = config
        self.bound_type = config.bound_type
        self.bound_order = config.bound_order

        if data_mean is not None:
            assert data_mean.shape == (1, 3, 32, 32), f"data_mean must be a 1x3x32x32 tensor, but got {data_mean.shape}"
        self.data_mean = data_mean

        assert config.embedding_type in ["positional"]
        assert config.encoder_type in ["standard", "skip", "residual"]
        assert config.decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = config.label_dropout
        emb_channels = config.model_channels * config.channel_mult_emb
        time_channels = config.model_channels * config.channel_mult_time
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=config.dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=config.resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_time = PositionalEmbedding(num_channels=time_channels, endpoint=False)
        self.map_label = Linear(in_features=config.label_dim, out_features=time_channels, **init) if config.label_dim else None
        self.map_augment = Linear(in_features=config.augment_dim, out_features=time_channels, bias=False, **init) if config.augment_dim else None
        self.map_layer0 = Linear(in_features=time_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = config.in_channels
        caux = config.in_channels
        for level, mult in enumerate(config.channel_mult):
            res = config.img_resolution >> level
            if level == 0:
                cin = cout
                cout = config.model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if config.encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=config.resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if config.encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=config.resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(config.num_blocks):
                cin = cout
                cout = config.model_channels * mult
                attn = (res in config.attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(config.channel_mult))):
            res = config.img_resolution >> level
            if level == len(config.channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(config.num_blocks + 1):
                cin = cout + skips.pop()
                cout = config.model_channels * mult
                attn = (idx == config.num_blocks and res in config.attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if config.decoder_type == 'skip' or level == 0:
                if config.decoder_type == 'skip' and level < len(config.channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=config.out_channels, out_channels=config.out_channels, kernel=0, up=True, resample_filter=config.resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=config.out_channels, kernel=3, **init_zero)

    def save_pretrained(self, save_directory: str, filename: str = "unet"):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, f"{filename}_config.json")
        config_dict = asdict(self.config)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
        print(f"Configuration saved to {config_path}")

        model_to_save = self.module if hasattr(self, "module") else self
        state_dict = model_to_save.state_dict()
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        output_model_file = os.path.join(save_directory, f"{filename}_model.pt")
        torch.save(state_dict_cpu, output_model_file)
        print(f"Model weights saved to {output_model_file}")

    @classmethod
    def from_pretrained(
        cls, save_directory: str, filename: str = "unet", use_ema: bool = False, data_mean: torch.Tensor = None
    ):
        config_path = os.path.join(save_directory, f"{filename}_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = cls.config_class(**config_dict)
        model = cls(config, data_mean=data_mean)

        if use_ema:
            model_path = os.path.join(save_directory, f"{filename}_ema.pt")
        else:
            model_path = os.path.join(save_directory, f"{filename}_model.pt")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")

        return model

    def _forward(self, x, time, class_labels=None, augment_labels=None):
        # Mapping.
        emb = self.map_time(time)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

    def forward(self, x, time, class_labels=None, augment_labels=None):
        x_t = x.clone()
        model_pred = self._forward(x, time, class_labels, augment_labels)
        t = match_dim_with_data(time, model_pred.shape, model_pred.device, model_pred.dtype, expand_dim=True)

        if self.bound_type == "none":
            return model_pred

        elif self.bound_type == "single":
            t = t ** self.bound_order
            return t * x_t + (1 - t) * model_pred
                
        elif self.bound_type == "double": # Be careful to use bound_order > 1 for the double boundary condition
            t = t ** self.bound_order
            return (
                (1 - t) * (self.data_mean - x_t) + 
                t * x_t + 
                t * (1. - t) * 4 * model_pred # bound_order > 1 will lead to unsymmetric coefficient
            )
        
        elif self.bound_type == "single_sin":
            return (
                torch.sin(torch.pi / 2. * t)**self.bound_order * x_t + 
                torch.cos(torch.pi / 2. * t)**self.bound_order * model_pred
            )

        elif self.bound_type == "double_sin":
            return (
                torch.cos(torch.pi / 2. * t)**self.bound_order * (self.data_mean - x_t) + 
                torch.sin(torch.pi / 2. * t)**self.bound_order * x_t + 
                torch.sin(torch.pi * t)**self.bound_order * model_pred
            )