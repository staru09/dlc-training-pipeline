#!/usr/bin/env python3
"""
Fix H5 file index format for DLC 3.0 compatibility.
Run this in the project directory before training.
"""

import pandas as pd
from pathlib import Path
import sys


def fix_h5_index(project_path: str):
    """Fix the H5 file index to match DLC 3.0 expected format."""
    project_path = Path(project_path)
    config_path = project_path / "config.yaml"
    
    if not config_path.exists():
        print(f"ERROR: config.yaml not found in {project_path}")
        return False
    
    # Find all labeled-data folders
    labeled_data_dir = project_path / "labeled-data"
    if not labeled_data_dir.exists():
        print(f"ERROR: labeled-data directory not found")
        return False
    
    for video_folder in labeled_data_dir.iterdir():
        if not video_folder.is_dir():
            continue
        
        # Find H5 files
        h5_files = list(video_folder.glob("CollectedData_*.h5"))
        csv_files = list(video_folder.glob("CollectedData_*.csv"))
        
        for h5_file in h5_files:
            print(f"\nProcessing: {h5_file}")
            
            try:
                # Read the H5 file
                df = pd.read_hdf(h5_file)
                print(f"  Current index type: {type(df.index)}")
                print(f"  Sample index: {df.index[0] if len(df.index) > 0 else 'empty'}")
                
                # Check if index needs fixing
                sample_idx = str(df.index[0]) if len(df.index) > 0 else ""
                
                if "labeled-data" not in sample_idx:
                    # Need to fix the index - prepend the path
                    video_name = video_folder.name
                    new_index = []
                    
                    for idx in df.index:
                        filename = str(idx)
                        # Create the expected path format
                        new_idx = f"labeled-data/{video_name}/{filename}"
                        new_index.append(new_idx)
                    
                    df.index = new_index
                    print(f"  Fixed index format")
                    print(f"  New sample index: {df.index[0]}")
                
                # Save the fixed H5 file
                df.to_hdf(h5_file, key="df_with_missing", mode="w")
                print(f"  ✓ Saved {h5_file.name}")
                
                # Also update the CSV for consistency
                csv_file = h5_file.with_suffix(".csv").with_name(
                    h5_file.name.replace(".h5", ".csv")
                )
                df.to_csv(csv_file)
                print(f"  ✓ Saved {csv_file.name}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                return False
    
    print("\n✓ All H5 files fixed!")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_h5.py <project_directory>")
        print("Example: python fix_h5.py dog_tracking-aru-2026-02-03")
        return
    
    project_path = sys.argv[1]
    fix_h5_index(project_path)


if __name__ == "__main__":
    main()
