import torch
import random
import numpy as np
import matplotlib.pyplot as plt


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