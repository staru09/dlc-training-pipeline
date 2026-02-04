#!/usr/bin/env python3
"""
Setup Roboflow dataset for DeepLabCut training.
Copies images from dataset folder to labeled-data folder and fixes H5/CSV indexes.
"""

import shutil
from pathlib import Path
import pandas as pd
import sys


def setup_dataset(project_path: str, dataset_path: str):
    """Copy images and fix annotations for DLC training."""
    
    project_path = Path(project_path)
    dataset_path = Path(dataset_path)
    
    # Find labeled-data folder
    labeled_data_dir = project_path / "labeled-data"
    if not labeled_data_dir.exists():
        print(f"ERROR: {labeled_data_dir} not found")
        return False
    
    # Get the video folder name (e.g., milo_15s) - prioritize folder with H5 files
    video_folders = list(labeled_data_dir.iterdir())
    video_folders = [f for f in video_folders if f.is_dir() and not f.name.startswith('.')]
    
    if not video_folders:
        print("ERROR: No video folders found in labeled-data")
        return False
    
    # Find the folder that has H5 files (the one with annotations)
    video_folder = None
    for folder in video_folders:
        h5_files = list(folder.glob("CollectedData_*.h5"))
        if h5_files:
            video_folder = folder
            break
    
    # Fallback: look for milo_15s specifically
    if video_folder is None:
        for folder in video_folders:
            if "milo" in folder.name.lower():
                video_folder = folder
                break
    
    # Last resort: first folder
    if video_folder is None:
        video_folder = video_folders[0]
    
    print(f"Target folder: {video_folder}")
    
    # Collect all images from train/valid/test
    all_images = {}
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            for img in split_dir.glob("*.jpg"):
                all_images[img.name] = img
            for img in split_dir.glob("*.png"):
                all_images[img.name] = img
    
    print(f"Found {len(all_images)} images in dataset")
    
    if len(all_images) == 0:
        print("ERROR: No images found in dataset folder")
        return False
    
    # Copy images to labeled-data folder
    print(f"\nCopying images to {video_folder.name}...")
    copied = 0
    for img_name, img_path in all_images.items():
        dest = video_folder / img_name
        if not dest.exists():
            shutil.copy2(img_path, dest)
            copied += 1
    
    print(f"Copied {copied} new images")
    
    # Now fix the H5 file index
    h5_files = list(video_folder.glob("CollectedData_*.h5"))
    csv_files = list(video_folder.glob("CollectedData_*.csv"))
    
    for h5_file in h5_files:
        print(f"\nFixing H5 index: {h5_file.name}")
        
        try:
            df = pd.read_hdf(h5_file)
            
            # Check current index format
            sample_idx = str(df.index[0]) if len(df.index) > 0 else ""
            print(f"  Current index sample: {sample_idx[:80]}...")
            
            # Build new index with correct path format
            new_index = []
            for idx in df.index:
                filename = str(idx)
                
                # Extract just the filename if it has a path prefix
                if "/" in filename:
                    filename = filename.split("/")[-1]
                
                # Build the DLC expected path format
                new_idx = f"labeled-data/{video_folder.name}/{filename}"
                new_index.append(new_idx)
            
            df.index = new_index
            
            # Save fixed H5
            df.to_hdf(h5_file, key="df_with_missing", mode="w")
            print(f"  ✓ Fixed {h5_file.name}")
            
            # Save corresponding CSV
            csv_file = h5_file.with_suffix(".csv")
            df.to_csv(csv_file)
            print(f"  ✓ Fixed {csv_file.name}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return False
    
    # Verify images exist
    print("\n--- Verification ---")
    if h5_files:
        df = pd.read_hdf(h5_files[0])
        missing = 0
        for idx in df.index[:10]:  # Check first 10
            img_path = project_path / idx
            if not img_path.exists():
                print(f"  MISSING: {idx}")
                missing += 1
            else:
                print(f"  ✓ Found: {idx}")
        
        if missing == 0:
            print("\n✓ All checked images exist!")
        else:
            print(f"\n⚠ {missing}/10 sample images are missing")
    
    print("\n✓ Dataset setup complete!")
    print(f"\nNow you can run:")
    print(f"  python train_only.py --config {project_path}/config.yaml --superanimal")
    
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python setup_dataset.py <project_directory> <dataset_directory>")
        print("Example: python setup_dataset.py dog_tracking-aru-2026-02-03 dataset")
        return
    
    project_path = sys.argv[1]
    dataset_path = sys.argv[2]
    setup_dataset(project_path, dataset_path)


if __name__ == "__main__":
    main()
