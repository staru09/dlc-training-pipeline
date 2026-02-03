"""
Configuration and constants for rf_to_dlc package.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Configuration settings for the rf_to_dlc pipeline."""

    # Dataset subfolders
    subfolders: List[str] = field(default_factory=lambda: ["train", "valid", "test"])

    # Video settings
    video_fps: int = 24
    video_codec: str = "mp4v"

    # Regex pattern to extract video name and frame number from filenames
    # Matches patterns like: "buzz_mp4-000042_jpg.rf.abc123.jpg"
    frame_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(r"(?P<video>.+?_mp4)-(?P<frame>\d+)_jpg")
    )

    # Keypoint visibility modes
    # 0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible
    VISIBILITY_VISIBLE_ONLY: int = 2  # Only include visible keypoints
    VISIBILITY_LABELED: int = 1  # Include all labeled keypoints (visible + occluded)

    # Default visibility mode
    visibility_mode: int = 2  # Default: only visible keypoints

    # Annotation file name
    annotation_filename: str = "_annotations.coco.json"

    # DLC labeled data prefix
    labeled_data_prefix: str = "labeled-data"

    # Default scorer name
    default_scorer: str = "j"

    # Default individual name for multi-animal projects
    default_individual: str = "individual1"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    def get_annotation_paths(self, base_dir: Path) -> List[Path]:
        """Get all annotation file paths from subfolders."""
        paths = []
        for sub in self.subfolders:
            ann_path = base_dir / sub / self.annotation_filename
            if ann_path.exists():
                paths.append(ann_path)
        return paths


# Default configuration instance
DEFAULT_CONFIG = Config()
