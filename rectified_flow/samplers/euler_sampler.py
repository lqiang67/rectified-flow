from .base_sampler import Sampler


class EulerSampler(Sampler):

    def step(self, **model_kwargs):
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)
        self.X_t = X_t + (t_next - t) * v_t