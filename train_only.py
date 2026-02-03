#!/usr/bin/env python3
"""
DLC Training Script (Dataset Already Prepared)
===============================================
Use this when you already have a DLC project with labeled data.
Skips project setup and goes straight to training.

Usage:
    python train_only.py --config path/to/config.yaml
    python train_only.py --config path/to/config.yaml --superanimal
    python train_only.py --config path/to/config.yaml --epochs 100 --batch-size 4
"""

import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Network settings
    "net_type": "dlcrnet_ms5",  # Options: dlcrnet_ms5, resnet_50, hrnet_w32
    "augmenter_type": "albumentations",
    
    # Training settings
    "epochs": 200,
    "batch_size": 8,
    "shuffle": 1,
    
    # SuperAnimal settings
    "superanimal_name": "superanimal_quadruped",
    
    # Evaluation
    "pcutoff": 0.6,
}


def create_dataset(config_path: str):
    """Create training dataset from labeled data."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 1: Creating Training Dataset")
    print("="*60)
    
    deeplabcut.create_training_dataset(
        config_path,
        num_shuffles=1,
        net_type=CONFIG["net_type"],
        augmenter_type=CONFIG["augmenter_type"],
    )
    
    print("✓ Training dataset created")


def train_network(config_path: str, use_superanimal: bool = False):
    """Train the network."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 2: Training Network")
    print("="*60)
    
    if use_superanimal:
        print("→ Using SuperAnimal transfer learning...")
        train_with_superanimal(config_path)
    else:
        print("→ Training from scratch...")
        print(f"  Network: {CONFIG['net_type']}")
        print(f"  Epochs: {CONFIG['epochs']}")
        print(f"  Batch size: {CONFIG['batch_size']}")
        
        deeplabcut.train_network(
            config_path,
            shuffle=CONFIG["shuffle"],
            maxiters=CONFIG["epochs"] * 1000,  # DLC uses iterations
            displayiters=500,
            saveiters=5000,
        )
    
    print("✓ Training complete")


def train_with_superanimal(config_path: str):
    """Train using SuperAnimal transfer learning."""
    import deeplabcut
    from deeplabcut.pose_estimation_pytorch import WeightInitialization
    
    superanimal_name = CONFIG["superanimal_name"]
    
    # Keypoint mapping for dogs
    project_to_super = {
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
    
    print("→ Creating keypoint mapping...")
    deeplabcut.modelzoo.create_conversion_table(
        config_path,
        super_animal=superanimal_name,
        project_to_super_animal=project_to_super,
    )
    print("✓ Keypoint mapping created")
    
    # Setup weight initialization
    print("→ Setting up weight initialization...")
    weight_init = WeightInitialization(
        dataset=superanimal_name,
        with_decoder=False,
    )
    
    # Create training dataset for SuperAnimal
    print("→ Creating training dataset...")
    deeplabcut.create_training_dataset(
        config_path,
        net_type="dlcrnet_ms5",
        augmenter_type="albumentations",
    )
    
    # Train with transfer learning
    print("→ Starting training with transfer learning...")
    print(f"  SuperAnimal: {superanimal_name}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    
    deeplabcut.train_network(
        config_path,
        shuffle=CONFIG["shuffle"],
        maxiters=CONFIG["epochs"] * 1000,
        displayiters=500,
        saveiters=5000,
        weight_init=weight_init,
    )


def evaluate_network(config_path: str):
    """Evaluate the trained network."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 3: Evaluating Network")
    print("="*60)
    
    deeplabcut.evaluate_network(
        config_path,
        Shuffles=[CONFIG["shuffle"]],
        plotting=True,
    )
    
    print("✓ Evaluation complete - check evaluation-results folder")


def analyze_videos(config_path: str, video_paths: list = None):
    """Analyze videos with trained model."""
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    
    print("\n" + "="*60)
    print("STEP 4: Analyzing Videos")
    print("="*60)
    
    if video_paths is None:
        # Get videos from config
        cfg = auxiliaryfunctions.read_config(config_path)
        video_paths = list(cfg.get("video_sets", {}).keys())
    
    if not video_paths:
        print("⚠ No videos to analyze. Skipping...")
        return
    
    deeplabcut.analyze_videos(
        config_path,
        video_paths,
        shuffle=CONFIG["shuffle"],
        save_as_csv=True,
    )
    
    print("✓ Video analysis complete")


def create_labeled_videos(config_path: str, video_paths: list = None):
    """Create labeled videos for visualization."""
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    
    print("\n" + "="*60)
    print("STEP 5: Creating Labeled Videos")
    print("="*60)
    
    if video_paths is None:
        cfg = auxiliaryfunctions.read_config(config_path)
        video_paths = list(cfg.get("video_sets", {}).keys())
    
    if not video_paths:
        print("⚠ No videos for labeling. Skipping...")
        return
    
    deeplabcut.create_labeled_video(
        config_path,
        video_paths,
        shuffle=CONFIG["shuffle"],
        filtered=False,
        pcutoff=CONFIG["pcutoff"],
        draw_skeleton=True,
    )
    
    print("✓ Labeled videos created")


def main():
    parser = argparse.ArgumentParser(
        description="Train DLC model (dataset already prepared)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_only.py --config dog_tracking-aru-2026-02-03/config.yaml
  python train_only.py --config config.yaml --superanimal
  python train_only.py --config config.yaml --epochs 100 --batch-size 4
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the DLC config.yaml file"
    )
    parser.add_argument(
        "--superanimal", "-s",
        action="store_true",
        help="Use SuperAnimal transfer learning"
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
        choices=["dlcrnet_ms5", "resnet_50", "hrnet_w32"],
        help="Network type (default: dlcrnet_ms5)"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset creation (if already created)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (evaluate existing model)"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Video paths for analysis (optional)"
    )
    
    args = parser.parse_args()
    
    # Update config
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["net_type"] = args.net_type
    
    # Validate config path
    config_path = str(Path(args.config).resolve())
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        return
    
    print("\n" + "="*60)
    print("  DLC TRAINING PIPELINE")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"SuperAnimal: {args.superanimal}")
    print(f"Network: {CONFIG['net_type']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    
    # Step 1: Create dataset (unless using superanimal or skipped)
    if not args.skip_dataset and not args.superanimal:
        create_dataset(config_path)
    
    # Step 2: Train
    if not args.skip_training:
        train_network(config_path, args.superanimal)
    
    # Step 3: Evaluate
    evaluate_network(config_path)
    
    # Step 4 & 5: Analyze and create labeled videos
    video_paths = args.videos
    analyze_videos(config_path, video_paths)
    create_labeled_videos(config_path, video_paths)
    
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nCheck the project folder for results:")
    print("  - evaluation-results/  : Model metrics and plots")
    print("  - dlc-models/          : Trained model snapshots")


if __name__ == "__main__":
    main()
