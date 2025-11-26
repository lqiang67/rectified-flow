import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from timm.models.vision_transformer import PatchEmbed, Mlp

from .lightningdit_utils import (
    VisionRotaryEmbeddingFast,
    SwiGLUFFN,
    RMSNorm,
    NormAttention,
    LabelEmbedder,
    get_2d_sincos_pos_embed,
    GaussianFourierEmbedding,
    modulate,
)

# ---------------------------
# Config
# ---------------------------

@dataclass
class LightningDiTConfig:
    input_size: int = 32
    patch_size: int = 1
    in_channels: int = 3
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 10
    use_qknorm: bool = False
    use_swiglu: bool = True
    use_rope: bool = True
    use_rmsnorm: bool = True
    wo_shift: bool = False
    time_scale: float = 1.0


# ---------------------------
# Blocks
# ---------------------------

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True, 
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LightningFinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------
# Model
# ---------------------------

class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config: LightningDiTConfig = None, 
    ):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads
        self.use_rope = config.use_rope
        self.use_rmsnorm = config.use_rmsnorm
        self.depth = config.depth
        self.hidden_size = config.hidden_size

        self.x_embedder = PatchEmbed(config.input_size, config.patch_size, config.in_channels, config.hidden_size, bias=True)
        self.t_embedder = GaussianFourierEmbedding(config.hidden_size, scale=config.time_scale)
        self.y_embedder = LabelEmbedder(config.num_classes, config.hidden_size, config.class_dropout_prob)
        self.ssl_supervise = False
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = config.hidden_size // config.num_heads // 2
            hw_seq_len = config.input_size // config.patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(config.hidden_size, 
                     config.num_heads, 
                     mlp_ratio=config.mlp_ratio, 
                     use_qknorm=config.use_qknorm, 
                     use_swiglu=config.use_swiglu, 
                     use_rmsnorm=config.use_rmsnorm,
                     wo_shift=config.wo_shift,
                     ) for _ in range(config.depth)
        ])
        self.final_layer = LightningFinalLayer(config.hidden_size, config.patch_size, self.out_channels, use_rmsnorm=config.use_rmsnorm)
        self.initialize_weights()

    def to_config(self) -> LightningDiTConfig:
        return self.config

    def save_pretrained(
        self,
        save_directory: str,
        filename: str = "lightningdit",
        state_dict = None,
        ema_state_dict = None,
    ):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, f"{filename}_config.json")
        config_dict = asdict(self.to_config())
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
        print(f"[LightningDiT] Configuration saved to {config_path}")
        model_to_save = self.module if hasattr(self, "module") else self
        if state_dict is None:
            state_dict = {k: v.cpu() for k, v in model_to_save.state_dict().items()}
        else:
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
        weights_path = os.path.join(save_directory, f"{filename}_model.pt")
        torch.save(state_dict, weights_path)
        print(f"[LightningDiT] Model weights saved to {weights_path}")

        if ema_state_dict is not None:
            ema_state_dict = {k: v.cpu() for k, v in ema_state_dict.items()}
            ema_path = os.path.join(save_directory, f"{filename}_ema.pt")
            torch.save(ema_state_dict, ema_path)
            print(f"[LightningDiT] EMA weights saved to {ema_path}")

    @classmethod
    def from_config(cls, config: LightningDiTConfig):
        return cls(**asdict(config))

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        filename: str = "lightningdit",
        use_ema: bool = False,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ):
        config_path = os.path.join(save_directory, f"{filename}_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = LightningDiTConfig(**config_dict)
        model = cls.from_config(config)

        model_path = os.path.join(
            save_directory, f"{filename}_ema.pt" if use_ema else f"{filename}_model.pt"
        )
        state_dict = torch.load(model_path, map_location=map_location)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if (missing or unexpected) and not strict:
            print(f"[LightningDiT.from_pretrained] missing keys: {missing}")
            print(f"[LightningDiT.from_pretrained] unexpected keys: {unexpected}")
        print(f"[LightningDiT] Model loaded from {model_path}")
        return model

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP (GaussianFourierEmbedding.mlp)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t=None, y=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        for block in self.blocks:
            x = block(x, c, feat_rope=self.feat_rope)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x
    
    def forward_with_cfg(self, x, t, y, cfg=1.0, 
                     schedule="constant", progress=None,
                     cfg_interval=None):
        """
        Supports 'constant' | 'linear' | 'interval'. Only supports synchronized timesteps.
        - 'linear': uses progress in [0,1] (defaults to 1 - t) to interpolate: eff = 1 + (cfg-1)*progress.
        - 'interval': apply CFG with `cfg` only when t is inside any (lo, hi) in `cfg_interval`;
                    outside the intervals use pure conditional (scale=1.0).
        """
        half = x[: len(x) // 2]
        x_combined = torch.cat([half, half], dim=0)
        out = self.forward(x_combined, t, y)
        v, rest = out[:, :self.in_channels], out[:, self.in_channels:]

        B2 = v.size(0) // 2
        cond_v, uncond_v = v[:B2], v[B2:]

        if schedule == "linear":
            if progress is None:
                t0 = float(t.flatten()[0].item()) if torch.is_tensor(t) else float(t)
                progress = 1.0 - max(0.0, min(1.0, t0))
            eff = 1.0 + (float(cfg) - 1.0) * float(max(0.0, min(1.0, progress)))
            guided_v = uncond_v + eff * (cond_v - uncond_v)

        elif schedule == "interval":
            if not cfg_interval:  # fallback to constant behavior
                guided_v = uncond_v + float(cfg) * (cond_v - uncond_v)
            else:
                t0 = float(t.flatten()[0].item()) if torch.is_tensor(t) else float(t)
                inside = any(lo <= t0 <= hi for (lo, hi) in cfg_interval)
                if inside:
                    guided_v = uncond_v + float(cfg) * (cond_v - uncond_v)  # apply cfg inside intervals
                else:
                    guided_v = cond_v  # scale=1.0 (pure conditional, no extra CFG boost)

        else:  # 'constant'
            guided_v = uncond_v + float(cfg) * (cond_v - uncond_v)

        v_guided = torch.cat([guided_v, guided_v], dim=0)
        return torch.cat([v_guided, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(-1e4, -1e4), interval_cfg: float = 0.0):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        half = x[: len(x) // 2] # cut the x by half, autoguidance does not need repeated input
        t = t[: len(t) // 2]
        y = y[: len(y) // 2]
        model_out = self.forward(half, t, y)
        ag_model_out = additional_model_forward(half, t, y)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]
        t = t[0]
        in_interval = False
        for i in range(len(cfg_interval)):
            if t >= cfg_interval[i][0] and t < cfg_interval[i][1]:
                if interval_cfg > 1.0:
                    eps = ag_eps + interval_cfg * (eps - ag_eps)
                in_interval = True
                break
        if not in_interval:
            eps = ag_eps + cfg_scale * (eps - ag_eps)
        return torch.cat([eps, eps], dim=0)


# ---------------------------
# Factory helpers & registry
# ---------------------------

def LightningDiT_XL(**kwargs):
    cfg = LightningDiTConfig(depth=28, hidden_size=1152, num_heads=16, **kwargs)
    return LightningDiT(cfg)

def LightningDiT_L(**kwargs):
    cfg = LightningDiTConfig(depth=24, hidden_size=1024, num_heads=16, **kwargs)
    return LightningDiT(cfg)

def LightningDiT_B(**kwargs):
    cfg = LightningDiTConfig(depth=12, hidden_size=768, num_heads=12, **kwargs)
    return LightningDiT(cfg)

def LightningDiT_S(**kwargs):
    cfg = LightningDiTConfig(depth=12, hidden_size=384, num_heads=6, **kwargs)
    return LightningDiT(cfg)


LightningDiT_models = {
    "LightningDiT-XL": LightningDiT_XL,  
    "LightningDiT-L":  LightningDiT_L, 
    "LightningDiT-B":  LightningDiT_B, 
    "LightningDiT-S":  LightningDiT_S,
}
