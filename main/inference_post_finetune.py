#!/usr/bin/env python3
"""
Post-Finetune Inference Script
==============================
Runs inference using a fine-tuned DLC model (via config.yaml).
This utilizes the weights learned during your training.

Usage:
    python inference_post_finetune.py
"""

import os
import argparse
from pathlib import Path
import deeplabcut

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your project's config.yaml
# You can leave this None and pass via CLI, or hardcode it here.
CONFIG_PATH = "path/to/project/config.yaml"

VIDEOS = [
    # Add your video paths here
    "videos/test_video.mp4",
]

SHUFFLE = 1
PCUTOFF = 0.6  # Confidence threshold for visualization

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned DLC model")
    parser.add_argument("--config", "-c", help="Path to DLC config.yaml", default=CONFIG_PATH)
    args = parser.parse_args()
    
    config_path = args.config
    
    print("="*60)
    print(f"POST-FINETUNE INFERENCE")
    print("="*60)
    
    if not config_path or not Path(config_path).exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        print("Please edit CONFIG_PATH in the script or pass --config argument.")
        return
        
    print(f"Config: {config_path}")
    
    # Resolve videos
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
    
    try:
        # Step 1: Analyze videos (get coordinates)
        print("\n→ Analyzing videos...")
        deeplabcut.analyze_videos(
            config_path,
            video_paths,
            shuffle=SHUFFLE,
            save_as_csv=True,
            # dynamic attributes could be added here if needed
        )
        
        # Step 2: Create labeled video
        print("\n→ Creating labeled videos...")
        deeplabcut.create_labeled_video(
            config_path,
            video_paths,
            shuffle=SHUFFLE,
            filtered=False,
            pcutoff=PCUTOFF,
            draw_skeleton=True,
        )
        
        print("\n✓ Inference complete!")
        print("Check for labeled videos in the same directory as your input videos.")
        
    except Exception as e:
        print(f"\n❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
