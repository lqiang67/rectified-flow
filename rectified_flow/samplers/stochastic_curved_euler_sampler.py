import torch
import warnings
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.flow_components.interpolation_solver import AffineInterp
from typing import Callable


class StochasticCurvedEulerSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        noise_replacement_rate: Callable | str = lambda t, t_next: 0.5,
        interp_inference: AffineInterp | str = "natural",
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )
        if isinstance(noise_replacement_rate, str):
            assert noise_replacement_rate.lower() == "ddpm", "currently only support ddpm"
            self.noise_replacement_rate = lambda t, t_next: 1 - (
                self.rectified_flow.interp.alpha(t) * self.rectified_flow.interp.beta(t_next) 
                / (self.rectified_flow.interp.alpha(t_next) * self.rectified_flow.interp.beta(t))
            )
        elif isinstance(noise_replacement_rate, Callable):
             self.noise_replacement_rate = noise_replacement_rate

        if isinstance(interp_inference, str):
            if interp_inference.lower() == "natural":
                self.interp_inference = self.rectified_flow.interp
            else:
                self.interp_inference = AffineInterp(interp_inference)
        elif isinstance(interp_inference, AffineInterp):
            self.interp_inference = interp_inference
        else:
            warnings.warn("It is only theoretically correct to use this sampler when pi0 is a zero mean Gaussian "
                          "and the coupling (X0, X1) is independent. Proceed at your own risk.")
        
        assert (
            self.rectified_flow.independent_coupling
            and self.rectified_flow.is_pi_0_zero_mean_gaussian
        ), "pi0 must be a zero mean gaussian and must use indepdent coupling"

    def step(self, **model_kwargs):
        """Perform a single step of the sampling process."""
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        dtype = v_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)

        # Given x_t and dot_x_t = vt, find the corresponding endpoints x_0 and x_1
        interp = self.interp_inference.solve(t, x_t=x_t, dot_x_t=v_t)
        x_1_pred = interp.x_1
        x_0_pred = interp.x_0

        # Randomize x_0_pred by replacing part of it with new noise
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        noise = noise.to(torch.float32)

        noise_replacement_factor = self.noise_replacement_rate(t, t_next)
        x_0_pred_refreshed = (
            (1 - noise_replacement_factor) * x_0_pred + 
            (1 - (1 - noise_replacement_factor) ** 2) **0.5 * noise
        )

        # Interpolate to find x_t at t_next
        self.x_t = self.rectified_flow.interp.solve(t_next, x_0=x_0_pred_refreshed, x_1=x_1_pred).x_t
        self.x_t = self.x_t.to(dtype)