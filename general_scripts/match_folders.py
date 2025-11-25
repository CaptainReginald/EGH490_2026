#!/usr/bin/env python3
"""
Script to ensure files in images folder have corresponding files in labels folder.
Removes extra files from the labels folder that don't have matching images.
Optionally filters train.txt to only include existing image files.
"""

import os
import argparse
from pathlib import Path


def get_file_stems(folder_path):
    """
    Get the file stems (filename without extension) from a folder.
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        set: Set of file stems
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return set()
    
    return {file.stem for file in folder.iterdir() if file.is_file()}


def filter_train_txt(train_txt_path, images_folder, dry_run=False):
    """
    Filter train.txt file to only include images that exist in the images folder.
    
    Args:
        train_txt_path (str): Path to train.txt file
        images_folder (str): Path to images folder
        dry_run (bool): If True, only show what would be changed without actually modifying
    """
    train_path = Path(train_txt_path)
    images_path = Path(images_folder)
    
    if not train_path.exists():
        print(f"Error: Train file {train_txt_path} does not exist")
        return
    
    if not images_path.exists():
        print(f"Error: Images folder {images_folder} does not exist")
        return
    
    # Read existing train.txt content
    with open(train_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} entries in train.txt")
    
    # Filter lines to only include existing images
    valid_lines = []
    removed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract filename from path (handle both relative and absolute paths)
        image_filename = Path(line).name
        image_path = images_path / image_filename
        
        if image_path.exists():
            valid_lines.append(line + '\n')
        else:
            removed_lines.append(line)
    
    print(f"Found {len(valid_lines)} valid image entries")
    print(f"Found {len(removed_lines)} entries for missing images")
    
    if removed_lines:
        print(f"\nEntries for missing images:")
        for removed in removed_lines:
            print(f"  - {removed}")
    
    if not dry_run and removed_lines:
        # Write filtered content back to train.txt
        with open(train_path, 'w') as f:
            f.writelines(valid_lines)
        print(f"\nUpdated train.txt - removed {len(removed_lines)} entries for missing images")
    elif dry_run and removed_lines:
        print(f"\nDry run: Would remove {len(removed_lines)} entries from train.txt")
    elif not removed_lines:
        print("\nTrain.txt is already clean - all entries have corresponding images!")


def remove_extra_labels(images_folder, labels_folder, dry_run=False):
    """
    Remove label files that don't have corresponding image files.
    
    Args:
        images_folder (str): Path to images folder
        labels_folder (str): Path to labels folder
        dry_run (bool): If True, only show what would be removed without actually removing
    """
    images_path = Path(images_folder)
    labels_path = Path(labels_folder)
    
    if not images_path.exists():
        print(f"Error: Images folder {images_folder} does not exist")
        return
    
    if not labels_path.exists():
        print(f"Error: Labels folder {labels_folder} does not exist")
        return
    
    # Get file stems from both folders
    image_stems = get_file_stems(images_folder)
    label_stems = get_file_stems(labels_folder)
    
    print(f"Found {len(image_stems)} image files")
    print(f"Found {len(label_stems)} label files")
    
    # Find extra label files (labels without corresponding images)
    extra_labels = label_stems - image_stems
    
    if not extra_labels:
        print("No extra label files found. Folders are already matched!")
        return
    
    print(f"\nFound {len(extra_labels)} extra label files:")
    
    removed_count = 0
    for label_stem in extra_labels:
        # Find the actual file with this stem in labels folder
        label_files = list(labels_path.glob(f"{label_stem}.*"))
        
        for label_file in label_files:
            print(f"  - {label_file.name}")
            
            if not dry_run:
                try:
                    label_file.unlink()
                    removed_count += 1
                    print(f"    Removed: {label_file}")
                except Exception as e:
                    print(f"    Error removing {label_file}: {e}")
            else:
                print(f"    Would remove: {label_file}")
    
    if dry_run:
        print(f"\nDry run completed. {len(extra_labels)} files would be removed.")
    else:
        print(f"\nRemoved {removed_count} extra label files.")
    
    # Report missing labels (images without corresponding labels)
    missing_labels = image_stems - label_stems
    if missing_labels:
        print(f"\nNote: Found {len(missing_labels)} image files without corresponding labels:")
        for missing in sorted(missing_labels):
            print(f"  - {missing}")


def main():
    images_folder = "/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/amil/cvat_export/amil_nov_b5af_subfolder02_250/data/images/train"
    labels_folder = "/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/amil/cvat_export/amil_nov_b5af_subfolder02_250/data/labels/train"
    train_txt_path = "/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/amil/cvat_export/amil_nov_b5af_subfolder02_250/train.txt"
    dry_run = False  # Set to True if you want to see what would be changed without making changes
    
    print(f"Images folder: {images_folder}")
    print(f"Labels folder: {labels_folder}")
    print(f"Train file: {train_txt_path}")
    
    if dry_run:
        print("Running in DRY RUN mode - no files will be removed")
    
    print("-" * 50)
    
    remove_extra_labels(images_folder, labels_folder, dry_run)
    
    print("\n" + "=" * 50)
    print("FILTERING TRAIN.TXT")
    print("=" * 50)
    filter_train_txt(train_txt_path, images_folder, dry_run)


if __name__ == "__main__":
    main()