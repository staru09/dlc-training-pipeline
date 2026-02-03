"""DLC project creation and management."""

from pathlib import Path
from typing import List, Optional, Union

import deeplabcut


def create_project(
    project_name: str,
    experimenter: str,
    videos: List[str] = None,
    working_directory: Optional[str] = None,
    copy_videos: bool = False,
    multianimal: bool = False,
) -> str:
    """Create a new DeepLabCut project.
    
    Args:
        project_name: Name of the project
        experimenter: Name of the experimenter/scorer
        videos: List of video paths (can be empty for pre-labeled data)
        working_directory: Where to create the project
        copy_videos: Whether to copy videos to project folder
        multianimal: Create multi-animal project
        
    Returns:
        Path to the created config.yaml
        
    Example:
        >>> config_path = create_project(
        ...     project_name="DogPose",
        ...     experimenter="john",
        ...     videos=["./videos/dog1.mp4"],
        ...     working_directory="./dlc_projects"
        ... )
    """
    if videos is None:
        videos = []
        
    config_path = deeplabcut.create_new_project(
        project=project_name,
        experimenter=experimenter,
        videos=videos,
        working_directory=working_directory,
        copy_videos=copy_videos,
        multianimal=multianimal,
    )
    
    return config_path


def check_labels(config_path: Union[str, Path]) -> None:
    """Check if labels are correctly placed on frames.
    
    Creates a subdirectory with the frames and overlaid labels
    for visual verification.
    
    Args:
        config_path: Path to project config.yaml
    """
    deeplabcut.check_labels(str(config_path))


def convert_csv_to_h5(
    config_path: Union[str, Path],
    scorer: Optional[str] = None,
) -> None:
    """Convert CollectedData CSV to HDF5 format.
    
    This must be done before creating a training dataset.
    
    Args:
        config_path: Path to project config.yaml
        scorer: Scorer name (uses config default if not specified)
    """
    kwargs = {"config": str(config_path)}
    if scorer:
        kwargs["scorer"] = scorer
        
    deeplabcut.convertcsv2h5(**kwargs)
