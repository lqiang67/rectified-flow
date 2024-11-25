import torch
import torch.nn.functional as F


class RectifiedFlowLossFunction:
    def __init__(self, loss_type: str = "mse"):
        """
        Initialize the loss function.

        Args:
            loss_type (str): Type of loss to use. Default is "mse".
        """
        self.loss_type = loss_type
    
    def __call__(
        self,
        v_t: torch.Tensor,
        dot_Xt: torch.Tensor,
        X_t: torch.Tensor,
        t: torch.Tensor,
        time_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss for rectified flow.

        Args:
            v_t (torch.Tensor): Predicted velocity tensor.
            dot_Xt (torch.Tensor): Ground truth velocity tensor.
            X_t (torch.Tensor): State tensor.
            t (torch.Tensor): Time tensor.
            time_weights (torch.Tensor): Weights for each time step.

        Returns:
            torch.Tensor: Computed loss value.
        """
        if self.loss_type == "mse":
            # Compute per-instance loss
            per_instance_loss = torch.mean((v_t - dot_Xt) ** 2, dim=list(range(1, v_t.dim())))
            # Weight the loss by time_weights and compute the mean
            loss = torch.mean(time_weights * per_instance_loss)
        else:
            raise NotImplementedError(f"Loss function '{self.loss_type}' is not implemented.")
        
        return loss