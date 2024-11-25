import torch
import random
import numpy as np

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.flow_components.utils import set_seed, match_dim_with_data


class Sampler:
    ODE_SAMPLING_STEP_LIMIT = 1000

    def __init__( # NOTE: consider using dataclass config
        self, 
        rectified_flow: RectifiedFlow,
        X_0: torch.Tensor | None = None, 
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        num_samples: int = 100,
        seed: int = 0, 
        callbacks: list[callable] | None = None
    ):
        self.seed = seed 
        set_seed(seed) 

        self.rectified_flow = rectified_flow
        self.num_samples = num_samples 

        # prepare 
        self.X_t = X_0 if X_0 is not None else self.rectified_flow.sample_noise(num_samples)
        self.X_0 = self.X_t.clone()
        self.num_steps, self.time_grid = self._prepare_time_grid(num_steps, time_grid)
        self.callbacks = callbacks or []
        self.record_traj_period = record_traj_period 
        self.step_count = 0
        self.time_iter = iter(self.time_grid)
        self.t = next(self.time_iter)
        self.t_next = next(self.time_iter)

        # recording trajectories 
        self._trajectories = [self.X_t.clone().cpu()]
        self._time_points = [self.t]    

    def _prepare_time_grid(self, num_steps, time_grid):
        if num_steps is None and time_grid is None:
            raise ValueError("At least one of num_steps or time_grid must be provided")

        if time_grid is None:
            time_grid = np.linspace(0, 1, num_steps + 1).tolist()
        else:
            if isinstance(time_grid, torch.Tensor):
                time_grid = time_grid.tolist()
            elif isinstance(time_grid, np.ndarray):
                time_grid = time_grid.tolist()
            elif not isinstance(time_grid, list):
                time_grid = list(time_grid)
            
            if num_steps is None:
                num_steps = len(time_grid) - 1
            else:
                assert len(time_grid) == num_steps + 1, "Time grid must have num_steps + 1 elements"

        return num_steps, time_grid

    def get_velocity(self, **model_kwargs): 
        X_t, t = self.X_t, self.t
        t = match_dim_with_data(t, X_t.shape, X_t.device, X_t.dtype, expand_dim=False)
        return self.rectified_flow.get_velocity(X_t, t, **model_kwargs)

    def step(self, **model_kwargs):
        """
        Performs a single integration step.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this step method.")

    def set_next_time_point(self):
        """Advances to the next time point."""
        self.step_count += 1
        try:
            self.t = self.t_next
            self.t_next = next(self.time_iter)
        except StopIteration:
            self.t_next = None

    def stop(self):
        """Determines whether the sampling should stop."""
        return (
            self.t_next is None
            or self.step_count >= self.ODE_SAMPLING_STEP_LIMIT
            or self.t >= 1.0 - 1e-6
        )

    def record(self):
        """Records trajectories and other information."""
        if self.step_count % self.record_traj_period == 0:
            self._trajectories.append(self.X_t.detach().clone().cpu())
            self._time_points.append(self.t)

            # Callbacks can be used for logging or additional recording
            for callback in self.callbacks:
                callback(self)

    @torch.inference_mode()
    def sample_loop(self, **model_kwargs):
        # NOTE: consider support multiple times of sampling
        """Runs the sampling process"""
        while not self.stop():
            self.step(**model_kwargs)
            self.record()
            self.set_next_time_point()
        return self

    @property
    def trajectories(self) -> list[torch.Tensor]:
        """List of recorded trajectories."""
        return self._trajectories

    @property
    def time_points(self) -> list[float]:
        """List of recorded time points."""
        return self._time_points
