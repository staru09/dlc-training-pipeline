"""
File management utilities for organizing images and data.
"""

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG, Config
from .utils import find_image_in_subfolders, parse_frame_info


def get_image_names_from_df(
    df: pd.DataFrame,
    header_rows: int = 3,
) -> List[str]:
    """
    Extract image filenames from a formatted DataFrame.

    Args:
        df: Combined DataFrame with headers
        header_rows: Number of header rows (3 for single-animal, 4 for multi-animal)

    Returns:
        List of image filenames (basename only)
    """
    image_paths = df.iloc[header_rows:, 0].tolist()
    return [Path(p).name for p in image_paths]


def copy_images_to_project(
    image_names: List[str],
    source_dir: Path,
    dest_dir: Path,
    subfolders: Optional[List[str]] = None,
) -> Tuple[int, List[str]]:
    """
    Copy images from source directories to DLC project folder.

    Args:
        image_names: List of image filenames to copy
        source_dir: Base directory containing source images
        dest_dir: Destination directory (DLC labeled-data folder)
        subfolders: List of subfolder names to search in source_dir

    Returns:
        Tuple of (number of files copied, list of missing files)
    """
    if subfolders is None:
        subfolders = DEFAULT_CONFIG.subfolders

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []

    for img_name in image_names:
        img_path = find_image_in_subfolders(img_name, source_dir, subfolders)
        if img_path:
            shutil.copy2(img_path, dest_dir / img_name)
            copied += 1
        else:
            missing.append(img_name)

    print(f"✅ Copied {copied} image(s) to {dest_dir}")
    if missing:
        print(f"⚠️ {len(missing)} image(s) NOT FOUND:")
        for m in missing[:10]:
            print(f"   - {m}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")

    return copied, missing


def create_frame_order_csv(
    base_dir: Path,
    output_path: Path,
    subfolders: Optional[List[str]] = None,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Create a CSV file mapping images to their video name and frame order.

    Args:
        base_dir: Base directory containing image subfolders
        output_path: Path for the output CSV file
        subfolders: List of subfolder names to search
        config: Configuration object

    Returns:
        DataFrame with columns [video_name, image_name, frame_number, subfolder]
    """
    if config is None:
        config = DEFAULT_CONFIG
    if subfolders is None:
        subfolders = config.subfolders

    base_dir = Path(base_dir)
    records = []

    for sub in subfolders:
        subfolder_path = base_dir / sub
        if not subfolder_path.exists():
            continue

        for img_path in subfolder_path.glob("*.jpg"):
            frame_info = parse_frame_info(img_path.name, config.frame_pattern)
            if frame_info:
                video_name, original_frame = frame_info
                records.append((video_name, img_path.name, original_frame, sub))

    # Create DataFrame
    df = pd.DataFrame(
        records, columns=["video_name", "image_name", "original_frame", "subfolder"]
    )

    # Sort and assign new frame numbers starting from 0 per video
    df = df.sort_values(by=["video_name", "original_frame"]).reset_index(drop=True)
    df["frame_number"] = df.groupby("video_name").cumcount()

    # Reorder columns
    df = df[["video_name", "image_name", "frame_number", "subfolder"]]

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Frame order CSV saved to: {output_path}")
    return df
