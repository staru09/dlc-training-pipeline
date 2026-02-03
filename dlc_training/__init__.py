"""
dlc_training - DLC Training Pipeline

A modular package for DeepLabCut model training workflows.
"""

from .config import TrainingConfig
from .project import (
    create_project,
    check_labels,
    convert_csv_to_h5,
)
from .dataset import (
    create_training_dataset,
    create_dataset_from_existing_split,
    create_training_comparison,
)
from .trainer import (
    train_network,
    finetune_network,
)
from .evaluator import (
    evaluate_network,
    evaluate_snapshots,
)
from .analyzer import (
    analyze_videos,
    extract_outlier_frames,
    refine_and_merge_labels,
)
from .visualizer import (
    create_labeled_video,
    plot_trajectories,
    create_labeled_video_manual,
)
from .superanimal import (
    create_keypoint_mapping,
    setup_weight_init,
    train_with_transfer_learning,
    train_with_memory_replay,
)
from .utils import (
    check_image_sizes,
    check_video_sizes,
    debug_annotation_data,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "TrainingConfig",
    # Project
    "create_project",
    "check_labels",
    "convert_csv_to_h5",
    # Dataset
    "create_training_dataset",
    "create_dataset_from_existing_split",
    "create_training_comparison",
    # Trainer
    "train_network",
    "finetune_network",
    # Evaluator
    "evaluate_network",
    "evaluate_snapshots",
    # Analyzer
    "analyze_videos",
    "extract_outlier_frames",
    "refine_and_merge_labels",
    # Visualizer
    "create_labeled_video",
    "plot_trajectories",
    "create_labeled_video_manual",
    # SuperAnimal
    "create_keypoint_mapping",
    "setup_weight_init",
    "train_with_transfer_learning",
    "train_with_memory_replay",
    # Utils
    "check_image_sizes",
    "check_video_sizes",
    "debug_annotation_data",
]
