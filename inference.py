#!/usr/bin/env python3
"""
Run inference on videos using a trained DeepLabCut model.

Usage:
    python inference.py --config path/to/config.yaml --video video.mp4
    python inference.py --config path/to/config.yaml --video videos/ --create-labeled
    python inference.py --config path/to/config.yaml --video video.mp4 --output results/
"""

import argparse
from pathlib import Path
import sys


def get_videos(video_path: Path) -> list:
    """Get list of video files from path (file or directory)."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if video_path.is_file():
        return [str(video_path)]
    elif video_path.is_dir():
        videos = []
        for ext in video_extensions:
            videos.extend([str(v) for v in video_path.glob(f"*{ext}")])
        return sorted(videos)
    else:
        return []


def run_inference(
    config_path: str,
    video_path: str,
    output_dir: str = None,
    create_labeled: bool = False,
    shuffle: int = 1,
    save_as_csv: bool = True,
    filter_predictions: bool = True
):
    """Analyze videos with trained DLC model."""
    
    print("=" * 60)
    print("  DLC INFERENCE")
    print("=" * 60)
    
    # Import DLC
    print("→ Loading DeepLabCut...")
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    
    print(f"  DLC version: {deeplabcut.__version__}")
    
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    
    video_path = Path(video_path)
    videos = get_videos(video_path)
    
    if not videos:
        print(f"ERROR: No videos found at {video_path}")
        sys.exit(1)
    
    print(f"\nConfig: {config_path}")
    print(f"Videos: {len(videos)}")
    for v in videos[:5]:
        print(f"  - {Path(v).name}")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more")
    
    # Determine output directory
    if output_dir:
        destfolder = str(Path(output_dir).resolve())
        Path(destfolder).mkdir(parents=True, exist_ok=True)
    else:
        destfolder = None  # DLC will save next to video
    
    # Step 1: Analyze videos
    print("\n" + "=" * 60)
    print("  STEP 1: Analyzing Videos")
    print("=" * 60)
    
    try:
        deeplabcut.analyze_videos(
            str(config_path),
            videos,
            shuffle=shuffle,
            save_as_csv=save_as_csv,
            destfolder=destfolder,
        )
        print("✓ Analysis complete!")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        raise
    
    # Step 2: Filter predictions (optional)
    if filter_predictions:
        print("\n" + "=" * 60)
        print("  STEP 2: Filtering Predictions")
        print("=" * 60)
        
        try:
            deeplabcut.filterpredictions(
                str(config_path),
                videos,
                shuffle=shuffle,
                destfolder=destfolder,
            )
            print("✓ Filtering complete!")
        except Exception as e:
            print(f"⚠ Filtering warning: {e}")
    
    # Step 3: Create labeled videos (optional)
    if create_labeled:
        print("\n" + "=" * 60)
        print("  STEP 3: Creating Labeled Videos")
        print("=" * 60)
        
        try:
            deeplabcut.create_labeled_video(
                str(config_path),
                videos,
                shuffle=shuffle,
                destfolder=destfolder,
                draw_skeleton=True,
                color_by="bodypart",
            )
            print("✓ Labeled videos created!")
        except Exception as e:
            print(f"✗ Video creation failed: {e}")
    
    # Step 4: Plot trajectories
    print("\n" + "=" * 60)
    print("  STEP 4: Plotting Trajectories")
    print("=" * 60)
    
    try:
        deeplabcut.plot_trajectories(
            str(config_path),
            videos,
            shuffle=shuffle,
            destfolder=destfolder,
        )
        print("✓ Trajectory plots saved!")
    except Exception as e:
        print(f"⚠ Plotting warning: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  INFERENCE COMPLETE")
    print("=" * 60)
    
    result_location = destfolder if destfolder else "same folder as videos"
    print(f"\nResults saved to: {result_location}")
    print("\nOutput files:")
    print("  - *_filtered.csv  : Smoothed keypoint predictions")
    print("  - *_filtered.h5   : Predictions in HDF5 format")
    if create_labeled:
        print("  - *_labeled.mp4   : Video with keypoints overlaid")
    print("  - *_trajectories.png : Keypoint movement plot")


def main():
    parser = argparse.ArgumentParser(
        description="Run DLC inference on videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --config project/config.yaml --video test.mp4
  python inference.py --config project/config.yaml --video videos/ --create-labeled
  python inference.py --config project/config.yaml --video test.mp4 --output results/
        """
    )
    parser.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    parser.add_argument("--video", "-v", required=True, help="Video file or folder")
    parser.add_argument("--output", "-o", help="Output directory (default: same as video)")
    parser.add_argument("--create-labeled", "-l", action="store_true", help="Create labeled video with keypoints")
    parser.add_argument("--shuffle", type=int, default=1, help="Shuffle index (default: 1)")
    parser.add_argument("--no-filter", action="store_true", help="Skip prediction filtering")
    parser.add_argument("--no-csv", action="store_true", help="Don't save CSV output")
    
    args = parser.parse_args()
    
    run_inference(
        config_path=args.config,
        video_path=args.video,
        output_dir=args.output,
        create_labeled=args.create_labeled,
        shuffle=args.shuffle,
        filter_predictions=not args.no_filter,
        save_as_csv=not args.no_csv
    )


if __name__ == "__main__":
    main()
