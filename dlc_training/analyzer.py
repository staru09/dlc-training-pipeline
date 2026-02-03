"""Video analysis functions."""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import deeplabcut


def analyze_videos(
    config_path: Union[str, Path],
    videos: List[str],
    shuffle: int = 1,
    save_as_csv: bool = True,
    videotype: str = ".mp4",
    batchsize: int = 8,
    dynamic_cropping: Optional[Dict[str, Any]] = None,
) -> None:
    """Analyze videos using a trained network.
    
    Results are stored as HDF5 files in the same directory as videos.
    
    Args:
        config_path: Path to project config.yaml
        videos: List of video paths or folder containing videos
        shuffle: Training shuffle index
        save_as_csv: Also save results as CSV
        videotype: Video file extension
        batchsize: Inference batch size
        dynamic_cropping: Dynamic cropping parameters for top-down models:
            - top_down_crop_size: (width, height)
            - patch_counts: (rows, cols)
            - patch_overlap: pixels
            - threshold: detection threshold
            - margin: crop margin
            
    Example:
        >>> analyze_videos(
        ...     config_path="./config.yaml",
        ...     videos=["./videos/dog_walking.mp4"],
        ...     shuffle=1
        ... )
    """
    kwargs = {
        "config": str(config_path),
        "videos": videos,
        "shuffle": shuffle,
        "save_as_csv": save_as_csv,
        "videotype": videotype,
    }
    
    if dynamic_cropping:
        kwargs["top_down_dynamic"] = dynamic_cropping
        kwargs["batchsize"] = 1  # Required for dynamic cropping
    else:
        kwargs["batchsize"] = batchsize
        
    deeplabcut.analyze_videos(**kwargs)


def extract_outlier_frames(
    config_path: Union[str, Path],
    videos: List[str],
    shuffle: int = 1,
    outlier_algorithm: str = "jump",
    p_bound: float = 0.01,
) -> None:
    """Extract frames where predictions are likely incorrect.
    
    Use this to identify frames that need manual correction.
    
    Args:
        config_path: Path to project config.yaml
        videos: Video paths to analyze
        shuffle: Training shuffle index
        outlier_algorithm: Algorithm for outlier detection ("jump", "fitting", etc.)
        p_bound: Probability threshold for outlier detection
        
    Example:
        >>> extract_outlier_frames(
        ...     config_path="./config.yaml",
        ...     videos=["./videos/dog_running.mp4"]
        ... )
    """
    deeplabcut.extract_outlier_frames(
        config=str(config_path),
        videos=videos,
        shuffle=shuffle,
        outlieralgorithm=outlier_algorithm,
        p_bound=p_bound,
    )


def refine_and_merge_labels(
    config_path: Union[str, Path],
) -> None:
    """Refine extracted outlier labels and merge with original dataset.
    
    This is a two-step process:
    1. Call deeplabcut.refine_labels() to manually adjust labels
    2. Call deeplabcut.merge_datasets() to add to training data
    
    Args:
        config_path: Path to project config.yaml
    """
    print("Step 1: Launching label refinement GUI...")
    print("Use napari or DLC GUI to adjust labels")
    deeplabcut.refine_labels(str(config_path))
    
    print("\nStep 2: Merging refined labels with dataset...")
    deeplabcut.merge_datasets(str(config_path))
    print("Done! Run create_training_dataset() to include new labels.")
