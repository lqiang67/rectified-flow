from .base_sampler import Sampler


class OverShootingSampler(Sampler):
    def __init__(self, c=1.0, overshooting_method='t + dt * (1 - t)', **kwargs):
        super().__init__(**kwargs)
        self.c = c

        # Define overshooting method
        if callable(overshooting_method):
            self.overshooting = overshooting_method
        elif isinstance(overshooting_method, str):
            try:
                self.overshooting = eval(f"lambda t, dt: {overshooting_method}")
            except SyntaxError:
                raise ValueError(f"Invalid overshooting method: {overshooting_method}")
        else:
            raise ValueError("Invalid overshooting method provided. Must be a string or callable.")

        # Ensure rf meets required conditions
        if not (self.rectified_flow.is_pi0_zero_mean_gaussian() and self.rectified_flow.independent_coupling):
            raise ValueError(
                "pi0 must be a zero-mean Gaussian distribution, and the coupling must be independent."
            )

    def step(self, **model_kwargs):
        """Perform a single overshooting step."""
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)
        alpha = self.rectified_flow.interp.alpha
        beta = self.rectified_flow.interp.beta

        # Calculate overshoot time and enforce constraints
        t_overshoot = min(self.overshooting(t_next, (t_next - t) * self.c), 1.0)
        if t_overshoot < t_next:
            raise ValueError("t_overshoot cannot be smaller than t_next.")

        # Advance to t_overshoot using ODE
        X_t_overshoot = X_t + (t_overshoot - t) * v_t

        # Apply noise to step back to t_next
        at = alpha(t_next) / alpha(t_overshoot)
        bt = (beta(t_next)**2 - (at * beta(t_overshoot))**2)**0.5
        noise = self.rectified_flow.sample_noise(self.num_samples)
        self.X_t = X_t_overshoot * at + noise * bt