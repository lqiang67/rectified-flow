import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import warnings

import torch.distributions as dist
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


def match_dim_with_data(
    t: torch.Tensor | float | List[float],
    X_shape: tuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    expand_dim: bool = True,
):
    """
    Prepares the time tensor by reshaping it to match the dimensions of X.

    Args:
        t (Union[torch.Tensor, float, List[float]]): Time tensor, which can be:
            - A scalar (float or 0-dimensional torch.Tensor)
            - A list of floats with length equal to the batch size or length 1
            - A torch.Tensor of shape (B,), (B, 1), or (1,)
        X_shape (tuple): Shape of the tensor X, e.g., X.shape

    Returns:
        torch.Tensor: Reshaped time tensor, ready for broadcasting with X.
    """
    B = X_shape[0]  # Batch size
    ndim = len(X_shape)

    if isinstance(t, float): # Create a tensor of shape (B,)
        t = torch.full((B,), t, device=device, dtype=dtype)
    elif isinstance(t, list):
        if len(t) == 1: # If t is a list of length 1, repeat the scalar value B times
            t = torch.full((B,), t[0], device=device, dtype=dtype)
        elif len(t) == B:
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            raise ValueError(f"Length of t list ({len(t)}) does not match batch size ({B}) and is not 1.")
    elif isinstance(t, torch.Tensor):
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0: # Scalar tensor, expand to (B,)
            t = t.repeat(B)
        elif t.ndim == 1:
            if t.shape[0] == 1: # Tensor of shape (1,), repeat to (B,)
                t = t.repeat(B)
            elif t.shape[0] == B: # t is already of shape (B,)
                pass
            else:
                raise ValueError(f"Batch size of t ({t.shape[0]}) does not match X ({B}).")
        elif t.ndim == 2:
            if t.shape == (B, 1): # t is of shape (B, 1), squeeze last dimension
                t = t.squeeze(1)
            elif t.shape == (1, 1): # t is of shape (1, 1), expand to (B,)
                t = t.squeeze().repeat(B)
            else:
                raise ValueError(f"t must be of shape ({B}, 1) or (1, 1), but got {t.shape}")
        else:
            raise ValueError(f"t can have at most 2 dimensions, but got {t.ndim}")
    else:
        raise TypeError(f"t must be a torch.Tensor, float, or a list of floats, but got {type(t)}.")

    # Reshape t to have singleton dimensions matching X_shape after the batch dimension
    if expand_dim:
        expanded_dims = [1] * (ndim - 1)
        t = t.view(B, *expanded_dims)

    return t


class AffineInterpSolver:
    """
    Symbolic solver for the equations:
        xt = at * x1 + bt * x0
        dot_xt = dot_at * x1 + dot_bt * x0
    Given known variables and unknowns set as None, the solver computes the unknowns.
    """
    def __init__(self):
        # Define symbols
        x0, x1, xt, dot_xt = sympy.symbols('x0 x1 xt dot_xt')
        at, bt, dot_at, dot_bt = sympy.symbols('at bt dot_at dot_bt')
        
        # Equations
        eq1 = sympy.Eq(xt, at * x1 + bt * x0)
        eq2 = sympy.Eq(dot_xt, dot_at * x1 + dot_bt * x0)
        
        # Variables to solve for
        variables = [x0, x1, xt, dot_xt]
        self.symbolic_solvers = {}
        
        # Create symbolic solvers for all pairs of unknown variables
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                unknown1, unknown2 = variables[i], variables[j]
                # print(f"Solving for {unknown1} and {unknown2}")
                # Solve equations
                solutions = sympy.solve([eq1, eq2], (unknown1, unknown2), dict=True)
                if solutions:
                    solution = solutions[0]
                    # Create lambdified functions
                    expr1 = solution[unknown1]
                    expr2 = solution[unknown2]
                    func = sympy.lambdify(
                        [x0, x1, xt, dot_xt, at, bt, dot_at, dot_bt],
                        [expr1, expr2],
                        modules="numpy"
                    )
                    # Store solver function
                    var_names = (str(unknown1), str(unknown2))
                    self.symbolic_solvers[var_names] = func

    def solve(self, results):
        known_vars = {k: getattr(results, k) for k in ['x0', 'x1', 'xt', 'dot_xt'] if getattr(results, k) is not None}
        unknown_vars = {k: getattr(results, k) for k in ['x0', 'x1', 'xt', 'dot_xt'] if getattr(results, k) is None}
        unknown_keys = tuple(unknown_vars.keys())

        if len(unknown_keys) > 2:
            raise ValueError("At most two variables among (x0, x1, xt, dot_xt) can be unknown.")
        elif len(unknown_keys) == 0:
            return results
        elif len(unknown_keys) == 1:
            # Select one known variable to make up the pair
            for var in ['x0', 'x1', 'xt', 'dot_xt']:
                if var in known_vars:
                    unknown_keys.append(var)
                    break

        func = self.symbolic_solvers.get(unknown_keys)

        # Prepare arguments in the order [x0, x1, xt, dot_xt, at, bt, dot_at, dot_bt]
        args = []
        for var in ['x0', 'x1', 'xt', 'dot_xt', 'at', 'bt', 'dot_at', 'dot_bt']:
            value = getattr(results, var, None)
            if value is None:
                value = 0  # Placeholder for unknowns
            args.append(value)

        # Compute the unknown variables
        solved_values = func(*args)
        # Assign the solved values back to results
        setattr(results, unknown_keys[0], solved_values[0])
        setattr(results, unknown_keys[1], solved_values[1])

        return results


class AffineInterp(nn.Module):
    def __init__(
        self, 
        name: str = "straight",
        alpha: Callable | None = None,
        beta: Callable | None = None,
        dot_alpha: Callable | None = None,
        dot_beta: Callable | None = None,
    ):
        super().__init__()

        if name.lower() in ['straight', 'lerp']:
            # Special case for "straight" interpolation
            alpha = lambda t: t
            beta = lambda t: 1 - t
            dot_alpha = lambda t: torch.ones_like(t)
            dot_beta = lambda t: -torch.ones_like(t)
            name = 'straight'
        elif name.lower() in ['harmonic', 'cos', 'sin', 'slerp', 'spherical']:
            # Special case of "spherical" interpolation
            alpha = lambda t: torch.sin(t * torch.pi / 2.0)
            beta = lambda t: torch.cos(t * torch.pi / 2.0)
            dot_alpha = lambda t: torch.cos(t * torch.pi / 2.0) * torch.pi / 2.0
            dot_beta = lambda t: -torch.sin(t * torch.pi / 2.0) * torch.pi / 2.0
            name = 'spherical'
        elif name.lower() in ['ddim', 'ddpm']:
            # DDIM/DDPM scheme; see Eq 7 in https://arxiv.org/pdf/2209.03003
            a = 19.9
            b = 0.1
            alpha = lambda t: torch.exp(-a * (1 - t) ** 2 / 4.0 - b * (1 - t) / 2.0)
            beta = lambda t: torch.sqrt(1 - alpha(t) ** 2)
            name = 'DDIM'
        elif alpha is not None and beta is not None and dot_alpha is not None and dot_beta is not None:
            # Custom interpolation functions
            raise NotImplementedError("Custom interpolation functions are not yet supported.")

        self.name = name
        self.alpha = lambda t: alpha(self.ensure_tensor(t))
        self.beta = lambda t: beta(self.ensure_tensor(t))
        if dot_alpha is not None: self.dot_alpha = lambda t: dot_alpha(self.ensure_tensor(t))
        else: self.dot_alpha = None 
        if dot_beta is not None: self.dot_beta = lambda t: dot_beta(self.ensure_tensor(t))
        else: self.dot_beta = None 

        self.solver = AffineInterpSolver()
        self.at = None
        self.bt = None
        self.dot_at = None
        self.dot_bt = None
        self.x0 = None
        self.x1 = None
        self.xt = None
        self.dot_xt = None

    @staticmethod
    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    @staticmethod
    def value_and_grad(f, input_tensor, detach=True):
        x = input_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            value = f(x)
            grad, = torch.autograd.grad(value.sum(), x, create_graph=not detach)
        if detach:
            value = value.detach()
            grad = grad.detach()
        return value, grad

    def get_coeffs(self, t, detach=True):
        if self.dot_alpha is None:
            at, dot_at = self.value_and_grad(self.alpha, t, detach=detach)
        else:
            at = self.alpha(t)
            dot_at = self.dot_alpha(t)
        if self.dot_beta is None:
            bt, dot_bt = self.value_and_grad(self.beta, t, detach=detach)
        else:
            bt = self.beta(t)
            dot_bt = self.dot_beta(t)
        self.at = at
        self.bt = bt
        self.dot_at = dot_at
        self.dot_bt = dot_bt
        return at, bt, dot_at, dot_bt

    def forward(self, X0, X1, t, detach=True):
        t = match_dim_with_data(t, X1.shape, device=X1.device, dtype=X1.dtype)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach=detach)
        Xt = a_t * X1 + b_t * X0
        dot_Xt = dot_a_t * X1 + dot_b_t * X0
        return Xt, dot_Xt

    def solve(self, t=None, x0=None, x1=None, xt=None, dot_xt=None):
        """
        Solve equation: xt = at*x1+bt*x0, dot_xt = dot_at*x1+dot_bt*x0
        Set any of the known variables, and keep the unknowns as None; the solver will fill the unknowns.
        Example: interp.solve(t, xt=xt, dot_xt=dot_xt); print(interp.x1.shape), print(interp.x0.shape)
        """
        if t is None:
            raise ValueError("t must be provided")
        self.x0 = x0
        self.x1 = x1
        self.xt = xt
        self.dot_xt = dot_xt
        x_not_none = next((v for v in [x0, x1, xt, dot_xt] if v is not None), None)
        if x_not_none is None:
            raise ValueError("At least two of x0, x1, xt, dot_xt must not be None")
        t = match_dim_with_data(t, x_not_none.shape, device=x_not_none.device, dtype=x_not_none.dtype)
        at, bt, dot_at, dot_bt = self.get_coeffs(t)
        self.solver.solve(self)


class TrainTimeSampler:
    def __init__(
        self,
        distribution: str = "uniform",
    ):
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample time tensor for training

        Returns:
            torch.Tensor: Time tensor, shape (batch_size,)
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,), device=device, dtype=dtype)
        else: # NOTE: will implement different time distributions
            raise NotImplementedError(f"Time distribution '{self.dist}' is not implemented.")
        
        return t


class TrainTimeWeights:
    def __init__(
        self,
        weight: str = "uniform",
    ):
        self.weight = weight

    def __call__(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")
        
        return wts


class RFLossFunction:
    def __init__(
        self,
        loss_type: str = "mse",
    ):
        self.loss_type = loss_type
    
    def __call__(
        self,
        v_t: torch.Tensor,
        dot_Xt: torch.Tensor,
        X_t: torch.Tensor,
        t: torch.Tensor,
        time_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            per_instance_loss = torch.mean((v_t - dot_Xt)**2, dim=list(range(1, v_t.dim())))
            loss = torch.mean(time_weights * per_instance_loss)
        else:
            raise NotImplementedError(f"Loss function '{self.loss_type}' is not implemented.")
        
        return loss


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
        criterion: RFLossFunction | str = "mse",
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
        self.criterion: RFLossFunction = (
            criterion if isinstance(criterion, RFLossFunction) else RFLossFunction(criterion)
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
        # SDE coeffs for dX = vt(Xt) + 0.5*sigma_t^2*Dlogp(Xt) + sigma_t*dWt
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
