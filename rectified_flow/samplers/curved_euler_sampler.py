from .base_sampler import Sampler

class CurvedEulerSampler(Sampler):
    def step(self, **model_kwargs):
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)

        self.rectified_flow.interp.solve(t, xt=X_t, dot_xt=v_t)
        X_1_pred = self.rectified_flow.interp.x1
        X_0_pred = self.rectified_flow.interp.x0

        # interplate to find x_{t_next}
        self.rectified_flow.interp.solve(t_next, x0=X_0_pred, x1=X_1_pred)
        self.X_t = self.rectified_flow.interp.xt