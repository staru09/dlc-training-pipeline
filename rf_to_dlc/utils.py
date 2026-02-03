"""
Shared utility functions for rf_to_dlc package.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .config import DEFAULT_CONFIG


def parse_frame_info(
    filename: str, pattern: Optional[re.Pattern] = None
) -> Optional[Tuple[str, int]]:
    """
    Extract video name and frame number from filename.

    Args:
        filename: Image filename to parse
        pattern: Regex pattern with 'video' and 'frame' groups.
                 Defaults to config pattern.

    Returns:
        Tuple of (video_name, frame_number) or None if no match
    """
    if pattern is None:
        pattern = DEFAULT_CONFIG.frame_pattern

    match = pattern.search(filename)
    if match:
        return match.group("video"), int(match.group("frame"))
    return None


def find_image_in_subfolders(
    image_name: str,
    base_dir: Path,
    subfolders: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Search for an image file across multiple subfolders.

    Args:
        image_name: Name of the image file to find
        base_dir: Base directory containing subfolders
        subfolders: List of subfolder names to search. Defaults to config subfolders.

    Returns:
        Path to the image if found, None otherwise
    """
    if subfolders is None:
        subfolders = DEFAULT_CONFIG.subfolders

    base_dir = Path(base_dir)
    for sub in subfolders:
        candidate = base_dir / sub / image_name
        if candidate.exists():
            return candidate
    return None


def get_keypoint_names_from_columns(columns: List[str]) -> List[str]:
    """
    Extract unique keypoint names from DataFrame column names.

    Args:
        columns: List of column names like ['nose_x', 'nose_y', 'eye_x', ...]

    Returns:
        Sorted list of unique keypoint names
    """
    keypoints = set()
    for col in columns:
        if col.endswith("_x"):
            keypoints.add(col[:-2])
        elif col.endswith("_y"):
            keypoints.add(col[:-2])
    return sorted(keypoints)


def clean_bodypart_names(bodyparts: List[str]) -> List[str]:
    """
    Clean bodypart names by removing duplicate suffixes like '.1'.

    Args:
        bodyparts: List of bodypart names

    Returns:
        Sorted list of clean bodypart names
    """
    return sorted({bp.replace(".1", "") for bp in bodyparts})


def extract_basename(path_str: str) -> str:
    """
    Extract the filename from a path string.

    Args:
        path_str: Path string (can include directory structure)

    Returns:
        Just the filename component
    """
    return Path(path_str).name
