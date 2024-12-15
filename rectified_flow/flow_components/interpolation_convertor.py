import torch
import copy
import warnings

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.flow_components.interpolation_solver import AffineInterp

class AffineInterpConverter:
    def __init__(
        self, 
        rectified_flow: RectifiedFlow,
        target_interp: AffineInterp,
    ):
        self.rectified_flow = rectified_flow
        self.source_interp = rectified_flow.interp
        self.target_interp = target_interp

    @staticmethod
    def binary_search_for_time(function, target: float, tol: float = 1e-6):
        """ Binary search for the time t such that function(t) = target, tensor version.
        function(t) is a monotonic function of t, of form function(t) = alpha(t) / beta(t).
        target: torch.Tensor, shape (batch_size,)
        function: takes in a tensor of shape (batch_size,) and returns a tensor of shape (batch_size,)
        """
        lower_bound = torch.zeros_like(target) + 1e-6
        upper_bound = torch.ones_like(target)

        mid_point = (upper_bound + lower_bound) / 2
        current_value = function(mid_point)
        iteration_count = 1

        while torch.max(torch.abs(current_value - target)) > tol and torch.max(torch.abs(upper_bound - lower_bound)) > tol:
            upper_bound = torch.where(current_value >= target, mid_point, upper_bound)
            lower_bound = torch.where(current_value <= target, mid_point, lower_bound)

            mid_point = (upper_bound + lower_bound) / 2.
            current_value = function(mid_point)
            iteration_count += 1

            if iteration_count > 1_000:
                warnings.warn("Binary search exceeded maximum iterations.", UserWarning)
                break

        return mid_point

    @staticmethod
    def match_time_and_scale(
        source_interp: AffineInterp, 
        target_interp: AffineInterp,
        target_time: torch.Tensor,
    ):
        """Given t' from target_interp, return corresponding t from source_interp and scaling factor."""
        source_ratio = lambda s: source_interp.alpha(s) / source_interp.beta(s)
        target_ratio = target_interp.alpha(target_time) / (target_interp.beta(target_time))
        raw_matched_t = AffineInterpConverter.binary_search_for_time(function=source_ratio, target=target_ratio)
        raw_scaling_factor = source_interp.alpha(raw_matched_t) / torch.max(target_interp.alpha(target_time), torch.full_like(target_time, 1e-12))

        # avoid numerical issues in the boundary
        tol = 1e-4
        close_to_zero = torch.abs(target_time - 0.0) < tol
        close_to_one = torch.abs(target_time - 1.0) < tol

        scaling_factor = raw_scaling_factor.clone()
        scaling_factor[close_to_zero] = 1.0
        scaling_factor[close_to_one] = 1.0

        matched_t = raw_matched_t.clone()
        matched_t[close_to_zero] = 0.0
        matched_t[close_to_one] = 1.0

        return matched_t, scaling_factor

    def get_transformed_velocity(
        self, 
        transformed_rf: RectifiedFlow,
        x_target: torch.Tensor,
        t_target: torch.Tensor,
    ):
        """Get the transformed velocity at x_target and time t_target."""
        # Match the time dimension with the data
        t_target = transformed_rf.match_dim_with_data(t_target, x_target.shape, expand_dim=False)
        t_source, scaling_factor = AffineInterpConverter.match_time_and_scale(self.source_interp, self.target_interp, t_target)
        scaling_factor = transformed_rf.match_dim_with_data(scaling_factor, x_target.shape, expand_dim=True)

        # Retrieve the original velocity and compute the transformed velocity
        x_source = scaling_factor * x_target
        original_velocity = self.rectified_flow.get_velocity(x_t=x_source, t=t_source)
        result = self.source_interp.solve(t=t_source, x_t=x_source, dot_x_t=original_velocity)

        return self.target_interp.solve(t=t_target, x_0=result.x_0, x_1=result.x_1).dot_x_t

    def transform_rectified_flow(self):
        """Transform the rectified flow to the target interpolation."""
        transformed_rf = copy.deepcopy(self.rectified_flow)
        transformed_rf.velocity_field = lambda x, t: self.get_transformed_velocity(transformed_rf, x, t)
        transformed_rf.interp = self.target_interp
        transformed_rf.source_interp = self.target_interp
        return transformed_rf