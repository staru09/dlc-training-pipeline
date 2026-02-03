"""Utility functions for debugging and diagnostics."""

import os
from pathlib import Path
from typing import Dict, List, Union
from collections import Counter

import cv2
import pandas as pd
from PIL import Image


def check_video_sizes(video_paths: List[Union[str, Path]]) -> Dict[str, tuple]:
    """Check frame dimensions of videos.
    
    Useful for debugging size mismatches between videos and training images.
    
    Args:
        video_paths: List of video file paths
        
    Returns:
        Dict mapping video path to (height, width, channels)
        
    Example:
        >>> sizes = check_video_sizes(["./video1.mp4", "./video2.mp4"])
        >>> for path, size in sizes.items():
        ...     print(f"{path}: {size}")
    """
    results = {}
    
    for video in video_paths:
        cap = cv2.VideoCapture(str(video))
        ret, frame = cap.read()
        if ret:
            results[str(video)] = frame.shape
        else:
            results[str(video)] = None
        cap.release()
        
    return results


def check_image_sizes(folder: Union[str, Path]) -> Dict[tuple, int]:
    """Check dimensions of all images in a folder.
    
    Scans recursively for image files and reports unique sizes.
    
    Args:
        folder: Path to folder containing images
        
    Returns:
        Dict mapping (width, height) to count
        
    Example:
        >>> sizes = check_image_sizes("./labeled-data/")
        >>> for size, count in sizes.items():
        ...     print(f"{size}: {count} images")
    """
    image_sizes = {}
    
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(subdir, file)
                try:
                    with Image.open(path) as img:
                        size = img.size  # (width, height)
                        image_sizes[path] = size
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
    # Count unique sizes
    counter = Counter(image_sizes.values())
    
    print("Unique image sizes and counts:")
    for size, count in counter.items():
        print(f"  {size}: {count} images")
        
    return dict(counter)


def debug_annotation_data(
    config_path: Union[str, Path],
    show_head: int = 5,
) -> pd.DataFrame:
    """Load and inspect merged annotation data.
    
    Useful for debugging dataset creation issues.
    
    Args:
        config_path: Path to project config.yaml
        show_head: Number of rows to display
        
    Returns:
        Merged annotation DataFrame
    """
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
    
    cfg = auxiliaryfunctions.read_config(str(config_path))
    
    trainingsetfolder = os.path.join(
        cfg["project_path"],
        auxiliaryfunctions.get_training_set_folder(cfg)
    )
    
    merged_df = merge_annotateddatasets(cfg, trainingsetfolder)
    
    print(f"Total annotations: {len(merged_df)}")
    print(f"\nColumns: {merged_df.columns.tolist()[:10]}...")
    print(f"\nHead ({show_head} rows):")
    print(merged_df.head(show_head))
    
    return merged_df


def get_bodyparts_from_config(config_path: Union[str, Path]) -> List[str]:
    """Extract bodyparts list from project config.
    
    Args:
        config_path: Path to project config.yaml
        
    Returns:
        List of bodypart names
    """
    from deeplabcut.utils import auxiliaryfunctions
    
    cfg = auxiliaryfunctions.read_config(str(config_path))
    
    if cfg.get("multianimalproject", False):
        from deeplabcut.utils import auxfun_multianimal
        _, uniquebodyparts, multianimalbodyparts = \
            auxfun_multianimal.extractindividualsandbodyparts(cfg)
        return multianimalbodyparts + uniquebodyparts
    else:
        return cfg["bodyparts"]


def parse_video_filenames(videos: List[str]) -> List[str]:
    """Parse video filenames from paths, removing duplicates.
    
    Args:
        videos: List of video paths
        
    Returns:
        Unique video filenames (without extension)
    """
    filenames = []
    seen = set()
    
    for video in videos:
        # Handle both Windows and Unix paths
        sep = "\\" if "\\" in video else "/"
        filename = video.rsplit(sep, 1)[-1]
        name, _ = os.path.splitext(filename)
        
        if name not in seen:
            filenames.append(name)
            seen.add(name)
            
    return filenames
