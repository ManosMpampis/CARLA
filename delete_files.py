#!/usr/bin/env python3
import os
import shutil
import torch
from pathlib import Path


def get_sorted_directories(base_path: str) -> list[Path]:
    """Get all subdirectories sorted by their numeric name."""
    base = Path(base_path)
    dirs = [d for d in base.iterdir() if d.is_dir()]
    # Sort by numeric value of directory name
    return sorted(dirs, key=lambda x: int(x.name))


def is_multiple_of_eval_epoch(name: str, eval_epoch: int = 50) -> bool:
    """Check if directory name is a multiple of 50."""
    try:
        num = int(name)
        return num % eval_epoch == 0
    except ValueError:
        return False


def get_last_non_multiple_of_eval_epoch(dirs: list[Path], eval_epoch:int = 50) -> Path | None:
    """Get the last directory that is NOT a multiple of 50."""
    non_multiples = [d for d in dirs if not is_multiple_of_eval_epoch(d.name, eval_epoch)]
    return non_multiples[-1] if non_multiples else None


def clean_directories(base_path: str, dry_run: bool = True, eval_every_n_epoch: int = 50) -> None:
    import re
    """
    Clean up directories keeping:
    - First 2 directories
    - The Last directory
    - The last non-multiple-of-eval directory
    - The directory of the epoch in the checkpoint
    
    Args:
        base_path: Path to the parent directory
        dry_run: If True, only print what would be deleted without actually deleting
    """
    epoch = torch.load(f"{base_path[:-12]}/checkpoint_best.pth.tar")["epoch"] if os.path.exists(f"{base_path[:-12]}/checkpoint_best.pth.tar") else 0
    base = Path(base_path)
    
    if not base.exists():
        return
    
    dirs = get_sorted_directories(base_path)
    
    if not dirs:
        return
    
    
    # Determine which directories to keep
    dirs_to_keep = set()
    
    # First 3 directories
    if len(dirs) >= 1:
        dirs_to_keep.add(dirs.pop(0).name)
    if len(dirs) >= 2:
        dirs_to_keep.add(dirs.pop(0).name)
    if len(dirs) >= 3:
        dirs_to_keep.add(dirs.pop(0).name)
    
    # Last Directory
    dirs_to_keep.add(dirs.pop(-1).name)
    
    # Last non-multiple of evaluation epoch
    last_non_multiple = get_last_non_multiple_of_eval_epoch(dirs, eval_every_n_epoch)
    if last_non_multiple:
        dirs_to_keep.add(last_non_multiple.name)
    # Directory of the checkpoint epoch
    padding_length = next((len(d) for d in dirs_to_keep if d and d.isdigit()), None)
    dirs_to_keep.add(str(epoch).zfill(padding_length))
    dirs_to_keep = list(dirs_to_keep)
    dirs_to_keep.sort()
    # Directories to delete
    dirs_to_delete = [d for d in dirs if d.name not in dirs_to_keep]
    

    dirs_to_write = {}
    for dir in dirs_to_keep:
        for d in Path(f"{base_path}/{dir}").iterdir():
            if str(d).rsplit("/", 1)[-1] not in dirs_to_write.keys():
                    dirs_to_write[str(d).rsplit("/", 1)[-1]] = []
            if Path(f"{base_path}/{dir}").is_dir():
                dirs_to_write[str(d).rsplit("/", 1)[-1]].append(dir)

    if dry_run:
        print(f"DRY RUN: {len(dirs_to_delete)} directories would be deleted.")
        print("Run with dry_run=False to actually delete them.")
    else:
        for d in dirs_to_delete:
            shutil.rmtree(d)
        try:
            embeddings_file_path = f"{base_path}/projector_config.pbtxt"
            with open(embeddings_file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"✗ Error: Embeddings file not found at '{embeddings_file_path}'")
            return

        output_lines = []
        for exp, nums in dirs_to_write.items():
            for num in nums:
                output_lines.append('embeddings {\n')
                output_lines.append(f'  tensor_name: "{exp}:{num}"\n')
                output_lines.append(f'  metadata_path: "{num}/{exp}/metadata.tsv"\n')
                output_lines.append(f'  tensor_path: "{num}/{exp}/tensors.tsv"\n')
                output_lines.append("}\n")

        with open(embeddings_file_path, 'w') as f:
            f.writelines(output_lines)


if __name__ == "__main__":
    import argparse
    experiment_cluster = ["layer", "instance", "batch"] #["origin", "model", "orig_big_batch"]
    directories = []
    # experiment_cluster = os.listdir(f'results/smd/')
    for exp_c in experiment_cluster:
        experiments = os.listdir(f'results/new_psm/{exp_c}')
        # experiments = ["small_eps_re_weight_initialization"]
        for exp in experiments:
            # machines = os.listdir(f'results/new_psm/{exp_c}/{exp}/')
            # machines = [file for file in machines if file.startswith('machine-')]
            machines = [""]
            for machine in machines:
                directories.append(f'results/new_psm/{exp_c}/{exp}/{machine}/pretext/tensorboard/')
                # shutil.rmtree(f'results/new_smd/{exp_c}/{exp}/{machine}/classification_entropy')
    

    for directory in directories:
        clean_directories(directory, dry_run=False)