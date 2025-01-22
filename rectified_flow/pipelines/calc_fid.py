import os
import csv
import re
from cleanfid import fid
import argparse

def parse_config_from_folder(folder_name: str):
    """
    根据文件夹名称解析 sampler 的关键信息。
    适配多种命名:
      - EulerSampler_steps16
      - DDPMSampler_steps32
      - OverShootingSampler_steps64_c2
      - SDESampler_steps64_noise_scale2.0_noise_decay_rate1.0
      - ...
    返回一个 dict, 包含以下字段:
      {
        "sampler": str,               # EulerSampler / DDPMSampler / OverShootingSampler / SDESampler
        "steps": int or None,
        "c_value": float or None,
        "noise_scale": float or None,
        "noise_decay_rate": float or None
      }
    """
    config_dict = {
        "sampler": None,
        "steps": None,
        "c_value": None,
        "noise_scale": None,
        "noise_decay_rate": None
    }

    # 1) sampler 名称 (xxSampler)
    sampler_match = re.search(r'([A-Za-z0-9]+Sampler)', folder_name)
    if sampler_match:
        config_dict["sampler"] = sampler_match.group(1)

    # 2) steps
    steps_match = re.search(r'steps(\d+)', folder_name)
    if steps_match:
        config_dict["steps"] = int(steps_match.group(1))

    # 3) c 值（仅适用于 OverShootingSampler，但如果存在就解析）
    c_match = re.search(r'_c([\d\.]+)', folder_name)
    if c_match:
        config_dict["c_value"] = float(c_match.group(1))

    # 4) noise_scale
    noise_scale_match = re.search(r'_noise_scale([\d\.]+)', folder_name)
    if noise_scale_match:
        config_dict["noise_scale"] = float(noise_scale_match.group(1))

    # 5) noise_decay_rate
    noise_decay_match = re.search(r'_noise_decay_rate([\d\.]+)', folder_name)
    if noise_decay_match:
        config_dict["noise_decay_rate"] = float(noise_decay_match.group(1))

    return config_dict


def main():
    """
    用 cleanfid 计算 base_dir 下所有子文件夹的 FID，并将结果记录到 csv_file。
    如果 csv_file 已经存在且包含对应文件夹记录，则跳过。
    """
    parser = argparse.ArgumentParser(description='FID calculation for all subfolders in a directory.')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Path to checkpoint directory, e.g. /root/autodl-tmp/unet_cifar_double_bound/checkpoint-500000')
    args = parser.parse_args()
    base_dir = args.base_dir
    csv_file = os.path.join(base_dir, "fid_results.csv")

    # 已有记录，避免重复计算
    existing_records = set()
    if os.path.exists(csv_file):
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder_name = row["folder_name"]
                existing_records.add(folder_name)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        file_is_empty = (os.path.getsize(csv_file) == 0)
        writer = csv.writer(f)

        # 如果文件是新建的，写表头
        if file_is_empty:
            writer.writerow([
                "folder_name",
                "sampler",
                "steps",
                "c_value",
                "noise_scale",
                "noise_decay_rate",
                "fid_score"
            ])

        # 2. 遍历所有子文件夹（images 文件夹）
        for folder_name in os.listdir(base_dir):
            subfolder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(subfolder_path):
                continue

            # 如果已有记录，则跳过
            if folder_name in existing_records:
                print(f"Skip folder: {folder_name}, already computed.")
                continue

            config_dict = parse_config_from_folder(folder_name)

            # 4. 计算 FID
            try:
                fid_score = fid.compute_fid(
                    subfolder_path,
                    dataset_name="cifar10",
                    dataset_res=32,
                    dataset_split="train",
                    mode="clean",
                    batch_size=100
                )
            except Exception as e:
                print(f"Error computing FID for folder: {folder_name}. Error: {e}")

            # 5. 将结果写入 CSV
            writer.writerow([
                folder_name,
                config_dict["sampler"],
                config_dict["steps"],
                config_dict["c_value"],
                config_dict["noise_scale"],
                config_dict["noise_decay_rate"],
                fid_score
            ])

            print(f"Folder: {folder_name} | FID: {fid_score:.4f}")

    print(f"\nAll FID results have been updated or saved to {csv_file}.")


if __name__ == "__main__":
    main()
