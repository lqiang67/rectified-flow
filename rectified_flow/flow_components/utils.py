import random
import torch
import matplotlib.pyplot as plt
import numpy as np


def match_dim_with_data(
    t: Union[torch.Tensor, float, List[float]],
    X_shape: tuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    expand_dim: bool = True,
) -> torch.Tensor:
    """
    Prepares the time tensor by reshaping it to match the dimensions of X.

    Args:
        t (Union[torch.Tensor, float, List[float]]): Time tensor, which can be:
            - A scalar (float or 0-dimensional torch.Tensor)
            - A list of floats with length equal to the batch size or length 1
            - A torch.Tensor of shape (B,), (B, 1), or (1,)
        X_shape (tuple): Shape of the tensor X, e.g., X.shape
        device (torch.device): The device for the output tensor.
        dtype (torch.dtype): The data type for the output tensor.
        expand_dim (bool): If True, reshape the output to match the dimensions of X.

    Returns:
        torch.Tensor: Reshaped time tensor, ready for broadcasting with X.
    """
    B = X_shape[0]  # Batch size
    ndim = len(X_shape)

    # Convert `t` to a tensor on the specified device and dtype
    if isinstance(t, float):
        t = torch.full((B,), t, device=device, dtype=dtype)
    elif isinstance(t, list):
        if len(t) == 1:
            t = torch.full((B,), t[0], device=device, dtype=dtype)
        elif len(t) == B:
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            raise ValueError(f"Length of t list ({len(t)}) does not match batch size ({B}) or is not 1.")
    elif isinstance(t, torch.Tensor):
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:  # Scalar tensor
            t = t.repeat(B)
        elif t.ndim == 1 and t.shape[0] == 1:  # Tensor of shape (1,)
            t = t.repeat(B)
        elif t.ndim == 2 and t.shape == (B, 1):  # Tensor of shape (B, 1)
            t = t.squeeze(1)
        elif t.shape[0] != B:  # Mismatched batch size
            raise ValueError(f"Batch size of t ({t.shape[0]}) does not match X ({B}).")
    else:
        raise TypeError(f"t must be a float, list, or torch.Tensor, but got {type(t)}.")

    # Expand dimensions to match X_shape
    if expand_dim:
        t = t.view(B, *([1] * (ndim - 1)))

    return t


def visualize_2d_trajectories(
    trajectories_list: list[torch.Tensor],
    D1_gt_samples: torch.Tensor = None,
    num_trajectories: int = 50,
    markersize: int = 3,
    dimensions: list[int] = [0, 1],
    alpha_trajectories: float = 0.5,
    alpha_generated_points: float = 1.0,
    alpha_gt_points: float = 1.0,
):
    """
    Plots 2D trajectories and points for visualization.
    
    Parameters:
        trajectories_list (list): List of trajectories to display.
        num_trajectories (int): Number of trajectories to display.
        markersize (int): Size of the markers. 
        dimensions (list): Indices of the dimensions to plot.
        alpha_trajectories (float): Transparency of trajectory lines.
        alpha_generated_points (float): Transparency of generated points.
        alpha_gt_points (float): Transparency of true points.
    """
    dim0, dim1 = dimensions
    D1_gt_samples = D1_gt_samples.clone().cpu().detach().numpy() if D1_gt_samples is not None else None
    traj_list_flat = [traj.clone().detach().cpu().reshape(traj.shape[0], -1) for traj in trajectories_list]

    xtraj = torch.stack(traj_list_flat)
    print("xtraj.shape", xtraj.shape)

    if D1_gt_samples is not None:
        plt.plot(D1_gt_samples[:, dim0], D1_gt_samples[:, dim1], '.', 
                    label='D1', markersize=markersize, alpha=alpha_gt_points)
    
    # Plot initial points from trajectories
    plt.plot(xtraj[0][:, dim0], xtraj[0][:, dim1], '.', 
                label='D0', markersize=markersize, alpha=alpha_gt_points)
    
    # Plot generated points
    plt.plot(xtraj[-1][:, dim0], xtraj[-1][:, dim1], 'r.', 
             label='Generated', markersize=markersize, alpha=alpha_generated_points)
    
    # Plot trajectory lines
    plt.plot(xtraj[:, :num_trajectories, dim0], xtraj[:, :num_trajectories, dim1], '--g', 
             alpha=alpha_trajectories)
    
    # Add legend and adjust layout
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False