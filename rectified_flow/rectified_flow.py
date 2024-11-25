import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import sympy
import warnings
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from rectified_flow.flow_components.interpolation import AffineInterp
from rectified_flow.flow_components.train_time_sampler import TrainTimeSampler
from rectified_flow.flow_components.train_time_weights import TrainTimeWeights
from rectified_flow.flow_components.loss_function import RectifiedFlowLossFunction
from rectified_flow.flow_components.utils import match_dim_with_data


class RectifiedFlow:
    def __init__(
        self,
        data_shape: tuple,
        model: nn.Module,
        interp: AffineInterp | str = "straight",
        prior_distribution: torch.distributions.Distribution | str = "normal",
        is_independent_coupling: bool = True,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        train_time_weight: TrainTimeWeights | str = "uniform",
        criterion: RectifiedFlowLossFunction | str = "mse",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        self.data_shape = data_shape
        self.model = model 

        self.interp: AffineInterp = (
            interp if isinstance(interp, AffineInterp) else AffineInterp(interp)
        )
        self.time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution)
        )
        self.train_time_weight: TrainTimeWeights = (
            train_time_weight
            if isinstance(train_time_weight, TrainTimeWeights)
            else TrainTimeWeights(train_time_weight)
        )
        self.criterion: RectifiedFlowLossFunction = (
            criterion if isinstance(criterion, RectifiedFlowLossFunction) else RectifiedFlowLossFunction(criterion)
        )

        self.pi_0 = prior_distribution if isinstance(prior_distribution, dist.Distribution) else dist.Normal(0, 1).expand(data_shape)
        self.independent_coupling = is_independent_coupling

        self.device = device
        self.dtype = dtype
        self.seed = seed

    def sample_time(self, batch_size: int):
        return self.time_sampler(batch_size, device=self.device, dtype=self.dtype)

    def sample_noise(self, batch_size: int):
        return self.pi_0.sample((batch_size,)).to(self.device, self.dtype)

    def get_interpolation(
        self,
        X_0: torch.Tensor,
        X_1: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Compute the interpolated values X_t and dot_Xt at time t

        Args:
            X_0 (torch.Tensor): X_0, shape (B, D) or (B, D1, D2, ..., Dn)
            X_1 (torch.Tensor): X_1, shape (B, D) or (B, D1, D2, ..., Dn)
            t (torch.Tensor): Time tensor, shape (B,), in [0, 1], no need for match_dim_with_data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: X_t, dot_Xt, both of shape (B, D) or (B, D1, D2, ..., Dn)
        """
        assert X_0.shape == X_1.shape, "X_0 and X_1 must have the same shape."
        assert X_0.shape[0] == X_1.shape[0], "Batch size of X_0 and X_1 must match."
        X_t, dot_Xt = self.interp.forward(X_0, X_1, t, detach=True)
        return X_t, dot_Xt

    def get_velocity(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        """
        Compute the velocity of the flow at (X_t, t)
        Decouples velocity computation from the model forward pass, handle t transformation, etc.

        Args:
            X (torch.Tensor): X_t, shape (B, D) or (B, D1, D2, ..., Dn)
            t (torch.Tensor): Time tensor, shape (B,), in [0, 1], no need for match_dim_with_data

        Returns:
            torch.Tensor: Velocity tensor, same shape as X
        """
        assert X.shape[0] == t.shape[0] and t.ndim == 1, "Batch size of X and t must match."
        # NOTE: May do t / velocity transformation, e.g. t = 1 - t, velocity = -velocity
        velocity = self.model(X, t, **kwargs)
        return velocity
    
    def get_loss(
        self,
        X_1: torch.Tensor,
        X_0: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Compute the loss of the flow model(X_t, t)

        Args:
            X_1 (torch.Tensor): X_1, shape (B, D) or (B, D1, D2, ..., Dn)
            X_0 (torch.Tensor): X_0, shape (B, D) or (B, D1, D2, ..., Dn), optional
            t (torch.Tensor): Time tensor, shape (B,), in [0, 1], optional
            **kwargs: Additional keyword arguments for the model input

        Returns:
            torch.Tensor: Loss tensor, scalar
        """
        if X_1.device != self.device:
            X_1 = X_1.to(self.device)
            warnings.warn("X_1 moved to device of the model.")

        if X_1.dtype != self.dtype:
            X_1 = X_1.to(self.dtype)
            warnings.warn("X_1 moved to dtype of the model.")

        t = self.sample_time(X_1.shape[0]) if t is None else t
        X_0 = self.sample_noise(X_1.shape[0]) if X_0 is None else X_0

        X_t, dot_Xt = self.get_interpolation(X_0, X_1, t)
        v_t = self.get_velocity(X_t, t, **kwargs)
        wts = self.train_time_weight(t)

        return self.criterion(v_t, dot_Xt, X_t, t, wts)

    def get_score_function_from_velocity(self, Xt, vt, t):
        # pi_0 (noise distribution) must ~ Normal(0,I), Dlogpt(Xt) = -E[X0 | Xt] / bt
        self.assert_canonical()
        self.interp.solve(t=t, xt=Xt, dot_xt=vt)
        dlogp = - self.interp.x0/self.interp.bt
        return dlogp

    def get_score_function(self, Xt, t, **kwargs):
        self.assert_canonical()
        vt = self.get_velocity(Xt, t, **kwargs)
        return self.get_score_function_from_velocity(Xt, vt, t)

    def get_sde_params_by_sigma(self, vt, xt, t, sigma):
        # SDE coeffs for dX = vt(Xt) + 0.5 * sigma_t^2 * Dlogp(Xt) + sigma_t * dWt
        self.assert_canonical()
        sigma_t = sigma(t)
        self.interp.solve(t=t, xt=xt, dot_xt=vt)
        dlogp = - self.interp.x0/self.interp.bt
        vt_sde = vt + 0.5 * sigma_t**2 * dlogp
        return vt_sde, sigma_t
    
    def get_stable_sde_params(self, vt, xt, t, e):
        # From SDE coeffs for dX = vt(Xt) - .5*sigma_t^2*E[X0|Xt]/bt + sigma_t*dWt,
        # let et^2 = sigmat^2/bt, we have sigmat = sqrt(bt) * et, we have:
        # dX = vt(Xt) - .5*et^2*E[X0|Xt]+ sqrt(bt) * et *dWt
        self.assert_canonical()
        self.interp.solve(t=t, xt=xt, dot_xt=vt)
        et = e(self.match_dim_with_data(t, xt.shape, device=xt.device, dtype=xt.dtype))
        x0_pred  = - self.interp.x0/self.interp.bt
        vt_sde = vt - x0_pred * et**2 * 0.5
        sigma_t = et * self.interp.bt**0.5
        # at, bt, dot_at, dot_bt = self.interp.get_coeffs(t)
        # vt_sde =vt * (1+et) - et * dot_at / at * xt
        # sigma_t_sde = (2 * (1-at) * dot_at/(at) * et)**(0.5)
        return vt_sde, sigma_t

    def is_pi0_zero_mean_gaussian(self):
        """Check if pi0 is a zero-mean Gaussian distribution."""
        is_multivariate_normal = (
            isinstance(self.pi_0, dist.MultivariateNormal) and 
            torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean))
        )
        is_normal = (
            isinstance(self.pi_0, dist.Normal) and 
            torch.allclose(self.pi_0.loc, torch.zeros_like(self.pi_0.loc))
        )
        return is_multivariate_normal or is_normal

    def match_dim_with_data(
        self,
        t: torch.Tensor | float | List[float],
        X_shape: tuple,
        expand_dim: bool = True,
    ):
        return match_dim_with_data(t, X_shape, device=self.device, dtype=self.dtype, expand_dim=expand_dim)

    def is_pi0_standard_gaussian(self):
        """Check if pi0 is a standard Gaussian distribution."""
        is_multivariate_normal = (
            isinstance(self.pi_0, dist.MultivariateNormal) and
            torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean)) and
            torch.allclose(
                self.pi_0.covariance_matrix,
                torch.eye(self.pi_0.mean.size(0), device=self.pi_0.mean.device)
            )
        )
        is_normal = (
            isinstance(self.pi_0, dist.Normal) and
            torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean)) and
            torch.allclose(self.pi_0.variance, torch.ones_like(self.pi_0.variance))
        )
        return is_multivariate_normal or is_normal

    def assert_pi0_is_standard_gaussian(self):
        """Raise an error if pi0 is not a standard Gaussian distribution."""
        if not self.is_pi0_standard_gaussian():
            raise ValueError("pi0 must be a standard Gaussian distribution.")

    def assert_canonical(self):
        """Raise an error if the distribution is not in canonical form."""
        if not (self.is_pi0_standard_gaussian() and self.independent_coupling):
            raise ValueError(
                "Must be the Canonical Case: pi0 must be a standard Gaussian "
                "and the data must be unpaired (independent coupling)."
            )
