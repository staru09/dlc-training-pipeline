#!/usr/bin/env python3
"""
Initialize a DeepLabCut project with custom configuration.

Usage:
    python init_project.py --name my_project --scorer aru --labeled-data labeled-data/milo_15s
    python init_project.py --name dog_tracking --scorer john --bodyparts config/bodyparts.txt
"""

import argparse
from pathlib import Path
from datetime import date


# Default dog keypoints (SuperAnimal quadruped compatible)
DEFAULT_BODYPARTS = [
    "right_eye", "left_eye", "nose", "top_head",
    "right_ear_base", "left_ear_base", "neck",
    "right_front_wrist", "right_front_paw",
    "left_front_wrist", "left_front_paw",
    "whithers", "spine_1", "spine_2", "tail_tip",
    "left_back_wrist", "left_back_paw",
    "right_back_wrist", "right_back_paw"
]

# Default skeleton connections
DEFAULT_SKELETON = [
    ["nose", "top_head"],
    ["top_head", "right_ear_base"],
    ["top_head", "left_ear_base"],
    ["top_head", "neck"],
    ["neck", "whithers"],
    ["whithers", "spine_1"],
    ["spine_1", "spine_2"],
    ["spine_2", "tail_tip"],
    ["neck", "right_front_wrist"],
    ["right_front_wrist", "right_front_paw"],
    ["neck", "left_front_wrist"],
    ["left_front_wrist", "left_front_paw"],
    ["spine_2", "left_back_wrist"],
    ["left_back_wrist", "left_back_paw"],
    ["spine_2", "right_back_wrist"],
    ["right_back_wrist", "right_back_paw"],
]


def create_project(
    project_name: str,
    scorer: str,
    labeled_data_path: Path,
    working_dir: Path = None,
    bodyparts: list = None,
    skeleton: list = None,
    net_type: str = "resnet_50"
) -> str:
    """Create a new DLC project with custom configuration."""
    
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    
    if bodyparts is None:
        bodyparts = DEFAULT_BODYPARTS
    if skeleton is None:
        skeleton = DEFAULT_SKELETON
    if working_dir is None:
        working_dir = Path.cwd()
    
    labeled_data_path = Path(labeled_data_path)
    working_dir = Path(working_dir)
    
    print("=" * 60)
    print("  INITIALIZE DLC PROJECT")
    print("=" * 60)
    print(f"Project: {project_name}")
    print(f"Scorer: {scorer}")
    print(f"Labeled data: {labeled_data_path}")
    print(f"Working dir: {working_dir}")
    print(f"Bodyparts: {len(bodyparts)}")
    print(f"Network: {net_type}")
    
    # Create project (need a dummy video path)
    today = date.today().strftime("%Y-%m-%d")
    project_folder = f"{project_name}-{scorer}-{today}"
    project_path = working_dir / project_folder
    
    # Create project structure manually if DLC create fails
    print("\n→ Creating project structure...")
    
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "labeled-data").mkdir(exist_ok=True)
    (project_path / "videos").mkdir(exist_ok=True)
    (project_path / "dlc-models").mkdir(exist_ok=True)
    (project_path / "training-datasets").mkdir(exist_ok=True)
    
    # Copy labeled data
    if labeled_data_path.exists():
        target_labeled = project_path / "labeled-data" / labeled_data_path.name
        if not target_labeled.exists():
            import shutil
            print(f"→ Copying labeled data to {target_labeled}...")
            shutil.copytree(labeled_data_path, target_labeled)
    
    # Create config.yaml
    config_path = project_path / "config.yaml"
    
    video_set_path = str(project_path / "labeled-data" / labeled_data_path.name)
    
    config_content = f"""# Project definitions
Task: {project_name}
scorer: {scorer}
date: {today.replace("-", "")}
multianimalproject: false
identity:

# Project path
project_path: {project_path}

# Default engine
engine: pytorch

# Annotation data
video_sets:
  {video_set_path}:
    crop: 0, 640, 0, 480

bodyparts:
{chr(10).join(f'- {bp}' for bp in bodyparts)}

# Frame extraction
start: 0
stop: 1
numframes2pick: 20

# Skeleton
skeleton:
{chr(10).join(f'- - {pair[0]}{chr(10)}  - {pair[1]}' for pair in skeleton)}

skeleton_color: black
pcutoff: 0.6
dotsize: 12
alphavalue: 0.7
colormap: rainbow

# Training configuration
TrainingFraction:
- 0.95
iteration: 0
default_net_type: {net_type}
default_augmenter: default
snapshotindex: -1
detector_snapshotindex: -1
batch_size: 8
detector_batch_size: 1

# Cropping
cropping: false
x1: 0
x2: 640
y1: 0
y2: 480

# Refinement
corner2move2:
- 50
- 50
move2corner: true

# SuperAnimal
SuperAnimalConversionTables:
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n✓ Created config: {config_path}")
    print(f"✓ Project created: {project_path}")
    
    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print(f"  1. Review/edit: {config_path}")
    print(f"  2. Train model: python train.py --config {config_path}")
    
    return str(config_path)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a DeepLabCut project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python init_project.py --name dog_tracking --scorer aru --labeled-data labeled-data/milo_15s
  python init_project.py --name cat_pose --scorer john --net-type resnet_101
        """
    )
    parser.add_argument("--name", "-n", required=True, help="Project name")
    parser.add_argument("--scorer", "-s", required=True, help="Scorer/experimenter name")
    parser.add_argument("--labeled-data", "-l", required=True, help="Path to labeled-data folder")
    parser.add_argument("--working-dir", "-w", default=".", help="Working directory (default: current)")
    parser.add_argument("--net-type", default="resnet_50", help="Network type (default: resnet_50)")
    parser.add_argument("--bodyparts", "-b", help="File with bodypart names (one per line)")
    
    args = parser.parse_args()
    
    # Load bodyparts from file if provided
    bodyparts = None
    if args.bodyparts:
        with open(args.bodyparts) as f:
            bodyparts = [line.strip() for line in f if line.strip()]
    
    create_project(
        project_name=args.name,
        scorer=args.scorer,
        labeled_data_path=Path(args.labeled_data),
        working_dir=Path(args.working_dir),
        bodyparts=bodyparts,
        net_type=args.net_type
    )


if __name__ == "__main__":
    main()
