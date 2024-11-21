from .base_sampler import Sampler


class SDESampler(Sampler):
    def __init__(self, e=lambda t: torch.ones_like(t), **kwargs):
        super().__init__(**kwargs)
        self.e = e

        if not (self.rectified_flow.is_pi0_standard_gaussian() and self.rectified_flow.independent_coupling):
            raise ValueError(
                "pi0 must be a standard Gaussian distribution, "
                "and the coupling must be independent."
            )

    def step(self, **model_kwargs):
        """Perform a single SDE sampling step."""
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)

        # Prepare time tensor and ensure it is within bounds
        t_ones = t * torch.ones((X_t.shape[0],), device=X_t.device)
        t_ones = match_dim_with_data(t_ones, X_t.shape, X_t.device, X_t.dtype, expand_dim=True)
        t_eps = 1e-12
        t_ones = torch.clamp(t_ones, t_eps, 1 - t_eps)
        step_size = t_next - t

        # Calculate alpha, beta, and their gradients
        e_t = self.e(t_ones)
        a_t, b_t, dot_a_t, dot_b_t = self.rectified_flow.interp.get_coeffs(t_ones)

        # Adjust velocity and calculate noise scale
        # print(f"e_t: {e_t.shape}, a_t: {a_t.shape}, b_t: {b_t.shape}, dot_a_t: {dot_a_t.shape}, dot_b_t: {dot_b_t.shape}")
        v_adj_t = (1 + e_t) * v_t - e_t * dot_a_t / a_t * X_t
        sigma_t = torch.sqrt(2 * (b_t**2 * dot_a_t / a_t - dot_b_t * b_t) * e_t)

        # Predict x1 and update xt with noise
        X_1_pred = X_t + (1 - t) * v_t
        noise = self.rectified_flow.sample_noise(self.num_samples)
        self.X_t = X_t + step_size * v_adj_t + sigma_t * step_size**0.5 * noise