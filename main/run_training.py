#!/usr/bin/env python3
"""
Complete DLC Training Pipeline Script
======================================
Run the entire training pipeline from CSV labels to trained model for multiple projects.

Usage:
    python run_training.py
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================

# Define your projects here
PROJECTS = [
    {
        "project_name": "project_1",
        "csv_path": "data.csv",
        "video_paths": ["videos/video1.mp4"],
    },
    # Add more projects as needed
    # {
    #     "project_name": "project_2",
    #     "csv_path": "path/to/other/data.csv",
    #     "video_paths": ["path/to/other/video.mp4"],
    # },
]

CONFIG = {
    "experimenter": "aru",
    
    # Network settings
    "net_type": "hrnet_w32",  # Enforced: hrnet_w32
    "augmenter_type": "albumentations",
    
    # Training settings
    "epochs": 200,
    "batch_size": 8,
    "shuffle": 1,
    
    # SuperAnimal settings (for transfer learning)
    "superanimal_name": "superanimal_quadruped",
    "use_superanimal": True, # Enforced: True
    
    # Evaluation settings
    "pcutoff": 0.6,  # Confidence threshold for visualization
}


def setup_project(project_config: dict, working_dir: str = "."):
    """Step 1: Create DLC project and import labels."""
    import deeplabcut
    
    print("\n" + "="*60)
    print(f"STEP 1: Setting Up Project: {project_config['project_name']}")
    print("="*60)
    
    # Create project
    project_name = project_config["project_name"]
    experimenter = CONFIG["experimenter"]
    video_paths = project_config["video_paths"]
    
    # Convert video paths to absolute strings
    video_paths = [str(Path(v).resolve()) for v in video_paths]

    try:
        config_path = deeplabcut.create_new_project(
            project=project_name,
            experimenter=experimenter,
            videos=video_paths,
            working_directory=working_dir,
            copy_videos=False
        )
    except Exception as e:
        print(f"WARNING: Project might already exist or creation failed: {e}")
        # Try to find existing config
        possible_path = Path(working_dir) / f"{project_name}-{experimenter}-{datetime.now().strftime('%Y-%m-%d')}" / "config.yaml"
        if possible_path.exists():
             config_path = str(possible_path)
        else:
             # Find most recent project folder matching name
             candidates = sorted(Path(working_dir).glob(f"{project_name}-{experimenter}-*"))
             if candidates:
                 config_path = str(candidates[-1] / "config.yaml")
             else:
                 raise e

    
    # Validate project was created
    if config_path == "nothingcreated" or not Path(config_path).exists():
        print("\n❌ ERROR: Project creation failed!")
        print("This usually means no valid video files were found or paths are incorrect.")
        raise SystemExit(1)
    
    print(f"✓ Created/Found project: {config_path}")
    
    # Import CSV labels to project
    import_labels_to_project(project_config["csv_path"], config_path)
    
    return config_path


def import_labels_to_project(csv_path: str, config_path: str):
    """Import CSV labels into the DLC project structure."""
    import pandas as pd
    from deeplabcut.utils import auxiliaryfunctions
    
    print("\n→ Importing labels from CSV...")
    
    # Read config
    cfg = auxiliaryfunctions.read_config(config_path)
    project_path = Path(config_path).parent
    
    # Read the CSV - handle different header formats
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    
    # Get unique video folders from the index
    video_folders = set()
    for idx in df.index:
        # Extract folder name from path like "labeled-data/video_name/frame.jpg"
        parts = str(idx).split("/")
        if len(parts) >= 2:
            video_folders.add(parts[1])
    
    print(f"  Found {len(video_folders)} video folder(s): {video_folders}")
    
    # Update config.yaml with bodyparts from CSV
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    cfg["bodyparts"] = bodyparts
    
    # Clear existing video_sets and add our labeled data folders as "videos"
    # This is a workaround for when actual video names don't match label folders
    cfg["video_sets"] = {}
    
    # Create labeled-data directories and save H5 files
    for video_folder in video_folders:
        labeled_data_dir = project_path / "labeled-data" / video_folder
        labeled_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter rows for this video
        video_df = df[df.index.str.contains(video_folder)]
        
        # Fix index to be proper format: just the image filename
        new_index = []
        for idx in video_df.index:
            # Get just the filename
            filename = str(idx).split("/")[-1]
            new_index.append(filename)
        video_df.index = new_index
        
        # Add this folder to video_sets in config (as a fake video path)
        # This tells DLC where to find the annotations
        fake_video_path = str(project_path / "labeled-data" / video_folder)
        cfg["video_sets"][fake_video_path] = {"crop": "0, 640, 0, 480"}
        
        # Save as H5 with proper multi-index structure
        h5_path = labeled_data_dir / f"CollectedData_{cfg['scorer']}.h5"
        video_df.to_hdf(h5_path, key="df_with_missing", mode="w")
        
        # Also save as CSV for reference
        csv_out_path = labeled_data_dir / f"CollectedData_{cfg['scorer']}.csv"
        video_df.to_csv(csv_out_path)
        
        print(f"  ✓ Saved {len(video_df)} frames to {labeled_data_dir.name}")
    
    # Write updated config
    auxiliaryfunctions.write_config(config_path, cfg)
    print(f"  Updated config with {len(bodyparts)} bodyparts")
    print("✓ Labels imported successfully")


def create_dataset(config_path: str):
    """Step 2: Create training dataset."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 2: Creating Training Dataset")
    print("="*60)
    
    # Ensure net_type is set in config (redundant but safe)
    # create_training_dataset actually uses the config.yaml usually, but passing net_type guarantees it
    deeplabcut.create_training_dataset(
        config_path,
        num_shuffles=1,
        net_type=CONFIG["net_type"],
        augmenter_type=CONFIG["augmenter_type"],
    )
    
    print("✓ Training dataset created")


def train_model(config_path: str):
    """Step 3: Train the network."""
    import deeplabcut
    
    print("\n" + "="*60)
    print("STEP 3: Training Network")
    print("="*60)
    
    if CONFIG["use_superanimal"]:
        print("→ Using SuperAnimal transfer learning...")
        train_with_superanimal(config_path)
    else:
        # Fallback (should not be reached with current config)
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
    from deeplabcut.core.engine import Engine

    superanimal_name = CONFIG["superanimal_name"]
    
    print(f"→ SuperAnimal: {superanimal_name}")
    print(f"→ Network: {CONFIG['net_type']}")

    # Create conversion table first
    project_to_super = create_default_keypoint_mapping()
    deeplabcut.modelzoo.create_conversion_table(
        config_path,
        super_animal=superanimal_name,
        project_to_super_animal=project_to_super,
    )
    print("✓ Keypoint mapping created")

    # Setup weight initialization for HRNet
    # This follows the pattern in train_model.py provided
    weight_init = WeightInitialization(
        dataset=superanimal_name,
        with_decoder=False, # Often False for transfer learning start
        # snapshot_path can be auto-resolved by DLC if dataset is provided, 
        # but specifying if known is good. For now, trusting DLC to fetch based on dataset name.
    )
    
    # Create training dataset specifying the weight init and net type explicitly
    # Note: create_training_dataset takes net_type. create_training_dataset_from_existing_split takes weight_init
    # We will use create_training_dataset_from_existing_split pattern if we want to mimic the notebook exactly,
    # but standard create_training_dataset should work if we pass the right params.
    # However, to be safe and follow the 'transfer learning' path:
    
    # We already called create_training_dataset in generic step 2. 
    # But for SuperAnimal, we often need to regen it with specific weight init in metadata? 
    # Actually, train_network accepts weight_init.
    
    deeplabcut.train_network(
        config_path,
        shuffle=CONFIG["shuffle"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        weight_init=weight_init,
        # Ensure we are using PyTorch engine
    )


def create_default_keypoint_mapping():
    """Create mapping from project keypoints to SuperAnimal-Quadruped."""
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
    
    print("✓ Evaluation complete")


def run_project_pipeline(project_config):
    """Run pipeline for a single project."""
    print(f"\nExample Project Processing: {project_config['project_name']}")
    
    # Step 1: Setup
    config_path = setup_project(project_config)
    
    # Step 2: Create Dataset
    create_dataset(config_path)
    
    # Step 3: Train
    train_model(config_path)
    
    # Step 4: Evaluate
    evaluate_model(config_path)
    
    return config_path


def main():
    print(f"Starting Multi-Project DLC Training Loop")
    print(f"Total Projects: {len(PROJECTS)}")
    print(f"Network: {CONFIG['net_type']}")
    print(f"SuperAnimal: {CONFIG['use_superanimal']}")
    
    for i, project in enumerate(PROJECTS):
        print("\n" + "*"*80)
        print(f"PROCESSING PROJECT {i+1}/{len(PROJECTS)}: {project['project_name']}")
        print("*"*80)
        
        try:
            run_project_pipeline(project)
        except Exception as e:
            print(f"❌ ERROR processing project {project['project_name']}: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next project...")


if __name__ == "__main__":
    main()
