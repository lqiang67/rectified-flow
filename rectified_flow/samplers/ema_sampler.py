import torch
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


def _match_norm_like(reference: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    batch = reference.shape[0]
    ref_norm = reference.reshape(batch, -1).norm(dim=1, keepdim=True)
    src_norm = source.reshape(batch, -1).norm(dim=1, keepdim=True)
    scale = (ref_norm / src_norm.clamp_min(torch.finfo(torch.float32).eps)).view(
        batch, *([1] * (reference.dim() - 1))
    )
    return source * scale


class EmaVelocityEulerSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int | None = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        guidance: float = 1.0,
        ema_decay: float = 0.9,
        preserve_v_norm: bool = False,
        unbiased_ema: bool = False,
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )
        self.guidance = float(guidance)
        self.ema_decay = float(ema_decay)
        self.preserve_v_norm = bool(preserve_v_norm)
        self.unbiased_ema = bool(unbiased_ema)
        self._ema_velocity: torch.Tensor | None = None
        self._ema_steps = 0

    def sample_loop(self, *args, **kwargs):
        self._ema_velocity = None
        self._ema_steps = 0
        return super().sample_loop(*args, **kwargs)

    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)

        dtype = x_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)

        if self._ema_velocity is None or self._ema_steps == 0:
            ema_velocity = v_t
            ema_velocity_next = (
                (1.0 - self.ema_decay) * v_t if self.unbiased_ema else v_t
            )
        else:
            if self.unbiased_ema:
                debias = 1.0 - self.ema_decay ** self._ema_steps
                ema_velocity = self._ema_velocity / max(
                    debias, torch.finfo(torch.float32).eps
                )
            else:
                ema_velocity = self._ema_velocity
            ema_velocity_next = (
                self.ema_decay * self._ema_velocity
                + (1.0 - self.ema_decay) * v_t
            )

        if self.preserve_v_norm:
            ema_velocity = _match_norm_like(v_t, ema_velocity)

        guided_velocity = self.guidance * v_t + (1.0 - self.guidance) * ema_velocity
        self.x_t = (x_t + (t_next - t) * guided_velocity).to(dtype)
        self._ema_velocity = ema_velocity_next.detach()
        self._ema_steps += 1
