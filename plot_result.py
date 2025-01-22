import pandas as pd
import matplotlib.pyplot as plt
import os

networks = {
    "unet_baseline": "UNet",
    "unet_single_boundary_order1": "UNet (right, t)",
    "unet_single_boundary_order2": "UNet (right, t^2)",
    "unet_double_boundary_order1": "UNet (double, t)",
    "unet_double_boundary_order2": "UNet (double, t^2)",
    "train_unet_double_bound_monotonic_mlp_cifar": "UNet (double, learned)",
}


for network_name, name in networks.items():

    # Define the directory containing your folders with fid_results.csv files
    root_dir = f'/checkpoint/storygen/lbo/boundary_models/{network_name}/checkpoint-200000/samples'

    csv_file_path = os.path.join(root_dir, 'fid_results.csv')
    df = pd.read_csv(csv_file_path)

    # Concatenate all dataframes into one
    df_all = df

    # Create a new column 'method' by combining 'sampler' and other hyperparameters
    def create_method_name(row):
        method = f"{row['sampler']}"
        if not pd.isnull(row['c_value']):
            method += f" c={row['c_value']}"
        elif not pd.isnull(row['noise_scale']):
            method += f" noise_scale={row['noise_scale']}"
            if not pd.isnull(row['noise_decay_rate']):
                method += f" noise_decay_rate={row['noise_decay_rate']}"
        return method

    df_all['method'] = df_all.apply(create_method_name, axis=1)

    # Plot FID score vs. steps for each method
    plt.figure(figsize=(8, 6))
    for method in df_all['method'].unique():
        df_method = df_all[df_all['method'] == method]
        df_method_sorted = df_method.sort_values(by='steps')  # Sort by 'steps'
        best_fid = df_method_sorted['fid_score'].min()
        plt.plot(df_method_sorted['steps'], df_method_sorted['fid_score'], label=f"{method} ({best_fid:.2f})", marker='o')
    plt.xlabel('Inference Steps')
    plt.ylabel('FID Score')
    plt.xscale('log')
    plt.xticks([16, 32, 64, 128, 256, 512, 1024])
    plt.gca().set_xticklabels(['16', '32', '64', '128', '256', '512', '1024'])  # Set custom labels
    plt.grid(axis='both', linestyle='--', color='lightgray')  # Add grid lines
    plt.title(name)
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Remove top frame
    ax.spines['right'].set_visible(False)  # Remove right frame
    plt.tight_layout()

    plt.savefig(f'./imgs/{name}.png')