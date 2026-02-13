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
from deeplabcut.modelzoo import video_inference_superanimal

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEOS = [
    # Add your video paths here
    "videos/video1.mp4",
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
            video_type=".mp4", # Assumes mp4, but DLC might detect others
        )
        print("\n✓ Inference complete!")
        print("Check for labeled videos in the same directory as your input videos.")
        
    except Exception as e:
        print(f"\n❌ ERROR during inference: {e}")

if __name__ == "__main__":
    main()
