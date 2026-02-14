#!/usr/bin/env python3
"""
Pre-Finetune Inference Script
=============================
Runs inference using the official SuperAnimal-Quadruped model (without local fine-tuning).
Use this to see how the base model performs on your videos.

Usage:
    python inference_pre_finetune.py
"""

import os
from pathlib import Path
import deeplabcut
from deeplabcut import video_inference_superanimal

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEOS = [
    # Add your video paths here
    "videos/video.mp4",
    # "videos/video2.mp4",
]

SUPERANIMAL_NAME = "superanimal_quadruped"
SCALE_LIST = range(200, 600, 50)  # Multi-scale inference settings

def main():
    print("="*60)
    print(f"PRE-FINETUNE INFERENCE: {SUPERANIMAL_NAME}")
    print("="*60)
    
    # Resolve video paths
    video_paths = []
    for v in VIDEOS:
        v_path = Path(v).resolve()
        if v_path.exists():
            video_paths.append(str(v_path))
        else:
            print(f"⚠ WARNING: Video not found: {v}")
    
    if not video_paths:
        print("❌ No valid videos found. Please check VIDEOS list in the script.")
        return

    print(f"Processing {len(video_paths)} videos...")
    
    # Run inference
    try:
        video_inference_superanimal(
            video_paths,
            SUPERANIMAL_NAME,
            scale_list=SCALE_LIST,
            videotype=".mp4", # Assumes mp4, but DLC might detect others
        )
        print("\n✓ Inference complete!")
        
        # Save results to CSV (DLC usually defaults to H5 or pickle)
        print("Converting outputs to CSV...")
        for video_path in video_paths:
            # DLC output filenames typically follow: VideoName + DLC_Model_Name + .h5
            # We look for files starting with the video stem in the same directory
            v_path = Path(video_path)
            output_dir = v_path.parent
            stem = v_path.stem
            
            # Find the H5 file generated for this video
            # Pattern: {video_name}DLC_{model_name}.h5
            # The model name might contain Shuffle, snapshot etc.
            # We'll rely on globbing for the specific pattern DLC uses
            candidates = list(output_dir.glob(f"{stem}*superanimal*.h5"))
            
            for h5_file in candidates:
                 try:
                     import pandas as pd
                     df = pd.read_hdf(h5_file)
                     csv_name = h5_file.with_suffix('.csv')
                     df.to_csv(csv_name)
                     print(f"   Saved CSV: {csv_name}")
                 except Exception as e:
                     print(f"   Failed to convert {h5_file} to CSV: {e}")

        print("Check for labeled videos and CSV files in the same directory as your input videos.")
        
    except Exception as e:
        print(f"\n❌ ERROR during inference: {e}")

if __name__ == "__main__":
    main()
