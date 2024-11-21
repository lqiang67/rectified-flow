from .base_sampler import Sampler


class NoiseRefreshSampler(Sampler):
    def __init__(self, noise_replacement_rate=lambda t: 0.5, **kwargs): 
        super().__init__(**kwargs)  
        self.noise_replacement_rate = noise_replacement_rate
        assert (self.rectified_flow.independent_coupling and self.rectified_flow.is_pi0_zero_mean_gaussian), \
            'pi0 must be a zero mean gaussian and must use indepdent coupling'

    def step(self, **model_kwargs):
        """Perform a single step of the sampling process."""
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)

        # Given xt and dot_xt = vt, find the corresponding endpoints x0 and x1
        self.rectified_flow.interp.solve(t, xt=X_t, dot_xt=v_t)
        X_1_pred = self.rectified_flow.interp.x1
        X_0_pred = self.rectified_flow.interp.x0

        # Randomize x0_pred by replacing part of it with new noise
        noise = self.rectified_flow.sample_noise(self.num_samples)
        noise_replacement_factor = self.noise_replacement_rate(t)
        X_0_pred_refreshed = (
            (1 - noise_replacement_factor**2)**0.5 * X_0_pred +
            noise * noise_replacement_factor
        )

        # Interpolate to find xt at t_next
        self.rectified_flow.interp.solve(t_next, x0=X_0_pred_refreshed, x1=X_1_pred)
        self.X_t = self.rectified_flow.interp.xt