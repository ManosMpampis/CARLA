#!/usr/bin/env python3
import os
import shutil
from pathlib import Path


def get_sorted_directories(base_path: str) -> list[Path]:
    """Get all subdirectories sorted by their numeric name."""
    base = Path(base_path)
    dirs = [d for d in base.iterdir() if d.is_dir()]
    # Sort by numeric value of directory name
    return sorted(dirs, key=lambda x: int(x.name))


def is_multiple_of_50(name: str) -> bool:
    """Check if directory name is a multiple of 50."""
    try:
        num = int(name)
        return num % 50 == 0
    except ValueError:
        return False


def get_last_non_multiple_of_50(dirs: list[Path]) -> Path | None:
    """Get the last directory that is NOT a multiple of 50."""
    non_multiples = [d for d in dirs if not is_multiple_of_50(d.name)]
    return non_multiples[-1] if non_multiples else None


def clean_directories(base_path: str, dry_run: bool = True) -> None:
    import re
    """
    Clean up directories keeping:
    - First 2 directories
    - The 0999 directory
    - The last non-multiple-of-50 directory
    
    Args:
        base_path: Path to the parent directory
        dry_run: If True, only print what would be deleted without actually deleting
    """
    base = Path(base_path)
    
    if not base.exists():
        # print(f"Error: Directory '{base_path}' does not exist.")
        return
    
    dirs = get_sorted_directories(base_path)
    
    if not dirs:
        # print("No directories found.")
        return
    
    # print(f"Found {len(dirs)} directories:\n")
    # for d in dirs:
    #     print(f"  {d.name}")
    # print()
    
    # Determine which directories to keep
    dirs_to_keep = set()
    
    # First 3 directories
    if len(dirs) >= 1:
        dirs_to_keep.add(dirs.pop(0).name)
    if len(dirs) >= 2:
        dirs_to_keep.add(dirs.pop(0).name)
    if len(dirs) >= 3:
        dirs_to_keep.add(dirs.pop(0).name)
    
    # Directory 0999
    dirs_to_keep.add(dirs.pop(-1).name)
    
    # Last non-multiple of 50
    last_non_multiple = get_last_non_multiple_of_50(dirs)
    if last_non_multiple:
        dirs_to_keep.add(last_non_multiple.name)
    
    # Directories to delete
    dirs_to_delete = [d for d in dirs if d.name not in dirs_to_keep]
    
    # print("Directories to KEEP:")
    # for name in sorted(dirs_to_keep):
        # print(f"  ✓ {name}")
    # print()
    
    # print("Directories to DELETE:")
    # for d in dirs_to_delete:
        # print(f"  ✗ {d.name}")
    # print()
    
    if dry_run:
        print(f"DRY RUN: {len(dirs_to_delete)} directories would be deleted.")
        print("Run with dry_run=False to actually delete them.")
    else:
        for d in dirs_to_delete:
            shutil.rmtree(d)
            # print(f"Deleted: {d.name}")
        # print(f"\nSuccessfully deleted {len(dirs_to_delete)} directories.")
            try:
                embeddings_file_path = f"{base_path}/projector_config.pbtxt"
                with open(embeddings_file_path, 'r') as f:
                    lines = f.readlines()
            except FileNotFoundError:
                print(f"✗ Error: Embeddings file not found at '{embeddings_file_path}'")
                return

            output_lines = []
            current_block = []
            in_block = False
            keep_block = False
            removed_count = 0

            for line in lines:
                stripped_line = line.strip()
                
                # Detect start of an embeddings block
                if stripped_line.startswith('embeddings {'):
                    in_block = True
                    keep_block = False
                    current_block = [line]
                    continue
                
                if in_block:
                    current_block.append(line)
                    
                    # Check if this block references a directory we're deleting
                    if 'tensor_name:' in line and 'Cluster:' in line:
                        match = re.search(r'tensor_name:\s*"Cluster:(\d+)"', line)
                        if match:
                            cluster_num = match.group(1)
                            if cluster_num in dirs_to_keep:
                                keep_block = True
                                removed_count += 1
                    
                    # Detect end of an embeddings block
                    if stripped_line == '}':
                        in_block = False
                        if keep_block:
                            output_lines.extend(current_block)
                        current_block = []
                
                else:
                    # Preserve lines that are not part of embeddings blocks
                    output_lines.append(line)

            # Write the filtered content back to the file
            with open(embeddings_file_path, 'w') as f:
                f.writelines(output_lines)


if __name__ == "__main__":
    import argparse
    experiment_cluster = ["layer_norm", "instance_norm", "batch_norm"] #["origin", "model", "orig_big_batch"]
    directories = []
    # experiment_cluster = os.listdir(f'results/smd/')
    for exp_c in experiment_cluster:
        # experiments = os.listdir(f'results/smd/{exp_c}')
        experiments = ["small_eps_re_weight_initialization"]
        for exp in experiments:
            machines = os.listdir(f'results/new_smd/{exp_c}/{exp}/')
            machines = [file for file in machines if file.startswith('machine-')]
            for machine in machines:
                directories.append(f'results/new_smd/{exp_c}/{exp}/{machine}/pretext/tensorboard/')
                # shutil.rmtree(f'results/new_smd/{exp_c}/{exp}/{machine}/classification_entropy')
    

    for directory in directories:
        clean_directories(directory, dry_run=False)