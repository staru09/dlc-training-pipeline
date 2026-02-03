#!/usr/bin/env python3
"""
Complete DLC Training Pipeline Script
======================================
Run the entire training pipeline from CSV labels to trained model.

Usage:
    python run_training.py --csv data.csv --videos /path/to/videos
    python run_training.py --csv data.csv --videos /path/to/videos --superanimal
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================

CONFIG = {
    # Project settings
    "project_name": "dog_tracking",
    "experimenter": "aru",
    
    # Network settings
    "net_type": "resnet_50",  # Options: dlcrnet_ms5, resnet_50, hrnet_w32
    "augmenter_type": "albumentations",
    
    # Training settings
    "epochs": 200,
    "batch_size": 8,
    "shuffle": 1,
    
    # SuperAnimal settings (for transfer learning)
    "superanimal_name": "superanimal_quadruped",
    
    # Evaluation settings
    "pcutoff": 0.6,  # Confidence threshold for visualization
}


def setup_project(csv_path: str, video_paths: list, working_dir: str = "."):
    """Step 1: Create DLC project and import labels."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 1: Setting Up Project")
    print("="*60)
    
    # Create project
    project_name = CONFIG["project_name"]
    experimenter = CONFIG["experimenter"]
    
    config_path = deeplabcut.create_new_project(
        project=project_name,
        experimenter=experimenter,
        videos=video_paths,
        working_directory=working_dir,
        copy_videos=False
    )
    
    # Validate project was created
    if config_path == "nothingcreated" or not Path(config_path).exists():
        print("\n❌ ERROR: Project creation failed!")
        print("This usually means no valid video files were found.")
        print("Please check that your video paths are correct and the files exist.")
        raise SystemExit(1)
    
    print(f"✓ Created project: {config_path}")
    
    # Import CSV labels to project
    import_labels_to_project(csv_path, config_path)
    
    return config_path


def import_labels_to_project(csv_path: str, config_path: str):
    """Import CSV labels into the DLC project structure."""
    import pandas as pd
    from deeplabcut.utils import auxiliaryfunctions
    
    print("\n→ Importing labels from CSV...")
    
    # Read config
    cfg = auxiliaryfunctions.read_config(config_path)
    project_path = Path(config_path).parent
    
    # Read the CSV
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    
    # Get unique video folders from the index
    video_folders = set()
    for idx in df.index:
        # Extract folder name from path like "labeled-data/video_name/frame.jpg"
        parts = idx.split("/")
        if len(parts) >= 2:
            video_folders.add(parts[1])
    
    print(f"  Found {len(video_folders)} video folder(s): {video_folders}")
    
    # Update config.yaml with bodyparts from CSV
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    cfg["bodyparts"] = bodyparts
    auxiliaryfunctions.write_config(config_path, cfg)
    print(f"  Updated config with {len(bodyparts)} bodyparts")
    
    # Create labeled-data directories and save H5 files
    for video_folder in video_folders:
        labeled_data_dir = project_path / "labeled-data" / video_folder
        labeled_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter rows for this video
        video_df = df[df.index.str.contains(video_folder)]
        
        # Fix index to just be image filenames
        video_df.index = [idx.split("/")[-1] for idx in video_df.index]
        
        # Save as H5
        h5_path = labeled_data_dir / f"CollectedData_{cfg['scorer']}.h5"
        video_df.to_hdf(h5_path, key="df_with_missing", mode="w")
        
        # Also save as CSV for reference
        csv_out_path = labeled_data_dir / f"CollectedData_{cfg['scorer']}.csv"
        video_df.to_csv(csv_out_path)
        
        print(f"  ✓ Saved {len(video_df)} frames to {labeled_data_dir.name}")
    
    print("✓ Labels imported successfully")


def create_dataset(config_path: str):
    """Step 2: Create training dataset."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 2: Creating Training Dataset")
    print("="*60)
    
    deeplabcut.create_training_dataset(
        config_path,
        num_shuffles=1,
        net_type=CONFIG["net_type"],
        augmenter_type=CONFIG["augmenter_type"],
    )
    
    print("✓ Training dataset created")


def train_model(config_path: str, use_superanimal: bool = False):
    """Step 3: Train the network."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 3: Training Network")
    print("="*60)
    
    if use_superanimal:
        print("→ Using SuperAnimal transfer learning...")
        train_with_superanimal(config_path)
    else:
        print("→ Training from scratch...")
        deeplabcut.train_network(
            config_path,
            shuffle=CONFIG["shuffle"],
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            displayiters=100,
            saveiters=1000,
        )
    
    print("✓ Training complete")


def train_with_superanimal(config_path: str):
    """Train using SuperAnimal transfer learning."""
    import deeplabcut
    from deeplabcut.modelzoo.utils import parse_available_supermodels
    from deeplabcut.pose_estimation_pytorch import WeightInitialization
    
    superanimal_name = CONFIG["superanimal_name"]
    
    # Create keypoint mapping (you may need to customize this)
    project_to_super = create_default_keypoint_mapping()
    
    deeplabcut.modelzoo.create_conversion_table(
        config_path,
        super_animal=superanimal_name,
        project_to_super_animal=project_to_super,
    )
    print("✓ Keypoint mapping created")
    
    # Setup weight initialization
    weight_init = WeightInitialization(
        dataset=superanimal_name,
        with_decoder=False,
    )
    
    # Train with transfer learning
    deeplabcut.create_training_dataset(
        config_path,
        net_type="dlcrnet_ms5",
    )
    
    deeplabcut.train_network(
        config_path,
        shuffle=CONFIG["shuffle"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        weight_init=weight_init,
    )


def create_default_keypoint_mapping():
    """Create mapping from project keypoints to SuperAnimal-Quadruped."""
    # Mapping for dog keypoints to SuperAnimal-Quadruped
    # Customize this based on your actual keypoints
    return {
        "right_eye": "right_eye",
        "left_eye": "left_eye",
        "nose": "nose",
        "top_head": "top_of_head",
        "right_ear_base": "right_earbase",
        "left_ear_base": "left_earbase",
        "neck": "throat",
        "right_front_wrist": "right_front_wrist",
        "right_front_paw": "right_front_paw",
        "left_front_wrist": "left_front_wrist",
        "left_front_paw": "left_front_paw",
        "whithers": "withers",
        "spine_1": "spine",
        "spine_2": "spine_mid",
        "tail_tip": "tail_end",
        "left_back_wrist": "left_back_wrist",
        "left_back_paw": "left_back_paw",
        "right_back_wrist": "right_back_wrist",
        "right_back_paw": "right_back_paw",
    }


def evaluate_model(config_path: str):
    """Step 4: Evaluate the trained network."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 4: Evaluating Network")
    print("="*60)
    
    deeplabcut.evaluate_network(
        config_path,
        Shuffles=[CONFIG["shuffle"]],
        plotting=True,
    )
    
    print("✓ Evaluation complete - check evaluation-results folder")


def analyze_videos(config_path: str, video_paths: list):
    """Step 5: Run inference on videos."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 5: Analyzing Videos")
    print("="*60)
    
    deeplabcut.analyze_videos(
        config_path,
        video_paths,
        shuffle=CONFIG["shuffle"],
        save_as_csv=True,
    )
    
    print("✓ Video analysis complete")


def create_labeled_videos(config_path: str, video_paths: list):
    """Step 6: Create labeled videos for visualization."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 6: Creating Labeled Videos")
    print("="*60)
    
    deeplabcut.create_labeled_video(
        config_path,
        video_paths,
        shuffle=CONFIG["shuffle"],
        filtered=False,
        pcutoff=CONFIG["pcutoff"],
        draw_skeleton=True,
    )
    
    print("✓ Labeled videos created")


def run_full_pipeline(csv_path: str, video_paths: list, use_superanimal: bool = False):
    """Run the complete training pipeline."""
    print("\n" + "="*60)
    print("  DLC TRAINING PIPELINE")
    print("="*60)
    print(f"CSV: {csv_path}")
    print(f"Videos: {video_paths}")
    print(f"SuperAnimal: {use_superanimal}")
    print(f"Network: {CONFIG['net_type']}")
    print(f"Epochs: {CONFIG['epochs']}")
    
    # Step 1: Setup project
    config_path = setup_project(csv_path, video_paths)
    
    # Step 2: Create training dataset
    create_dataset(config_path)
    
    # Step 3: Train model
    train_model(config_path, use_superanimal)
    
    # Step 4: Evaluate
    evaluate_model(config_path)
    
    # Step 5: Analyze videos
    analyze_videos(config_path, video_paths)
    
    # Step 6: Create labeled videos
    create_labeled_videos(config_path, video_paths)
    
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nProject config: {config_path}")
    print("Check the project folder for:")
    print("  - evaluation-results/  : Model evaluation metrics")
    print("  - videos/              : Analyzed videos with predictions")
    print("  - dlc-models/          : Trained model snapshots")
    
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Run complete DLC training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py --csv data.csv --videos ./videos/
  python run_training.py --csv data.csv --videos video1.mp4 video2.mp4 --superanimal
  python run_training.py --csv data.csv --videos ./videos/ --epochs 100 --batch-size 4
        """
    )
    
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to the labeled data CSV file"
    )
    parser.add_argument(
        "--videos", "-v",
        nargs="+",
        required=True,
        help="Path(s) to video files or directory containing videos"
    )
    parser.add_argument(
        "--superanimal", "-s",
        action="store_true",
        help="Use SuperAnimal transfer learning (recommended for dogs)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--net-type", "-n",
        default="dlcrnet_ms5",
        choices=["dlcrnet_ms5", "resnet_50", "hrnet_w32", "ctd_prenet_rtmpose_x"],
        help="Network architecture (default: dlcrnet_ms5)"
    )
    
    args = parser.parse_args()
    
    # Update config with CLI args
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["net_type"] = args.net_type
    
    # Expand video paths - convert to absolute paths
    video_paths = []
    for v in args.videos:
        v_path = Path(v).resolve()  # Convert to absolute path
        if v_path.is_dir():
            # Get all video files from directory
            for ext in [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]:
                for video_file in v_path.glob(f"*{ext}"):
                    video_paths.append(str(video_file.resolve()))
        elif v_path.is_file():
            video_paths.append(str(v_path))
        else:
            print(f"WARNING: Path not found: {v}")
    
    if not video_paths:
        print("ERROR: No video files found!")
        print("Make sure to provide paths to actual video files (.mp4, .avi, .mov, .mkv)")
        return
    
    print(f"Found {len(video_paths)} video(s):")
    for vp in video_paths:
        print(f"  - {vp}")
    
    # Run pipeline
    run_full_pipeline(args.csv, video_paths, args.superanimal)


if __name__ == "__main__":
    main()
