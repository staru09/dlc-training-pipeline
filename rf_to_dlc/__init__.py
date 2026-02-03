"""
rf_to_dlc: Roboflow to DeepLabCut Conversion Package

A modular toolkit for converting Roboflow annotated datasets
to DeepLabCut-compatible format for pose estimation training.
"""

from .config import Config
from .downloader import download_dataset, extract_dataset, download_and_extract
from .video_creator import group_frames_by_video, create_video_from_frames, create_all_videos
from .annotation_parser import (
    load_coco_annotations,
    extract_keypoints,
    parse_all_annotations,
    find_missing_keypoints,
)
from .data_formatter import (
    create_single_animal_headers,
    create_multi_animal_headers,
    format_dataframe,
    remove_duplicates,
    save_collected_data,
    convert_to_hdf5,
)
from .file_manager import (
    copy_images_to_project,
    create_frame_order_csv,
    get_image_names_from_df,
)
from .visualizer import load_collected_data, sample_and_visualize, verify_keypoint_mapping
from .utils import parse_frame_info, find_image_in_subfolders, get_keypoint_names_from_columns

__version__ = "1.0.0"
__all__ = [
    # Config
    "Config",
    # Downloader
    "download_dataset",
    "extract_dataset",
    "download_and_extract",
    # Video creator
    "group_frames_by_video",
    "create_video_from_frames",
    "create_all_videos",
    # Annotation parser
    "load_coco_annotations",
    "extract_keypoints",
    "parse_all_annotations",
    "find_missing_keypoints",
    # Data formatter
    "create_single_animal_headers",
    "create_multi_animal_headers",
    "format_dataframe",
    "remove_duplicates",
    "save_collected_data",
    "convert_to_hdf5",
    # File manager
    "copy_images_to_project",
    "create_frame_order_csv",
    "get_image_names_from_df",
    # Visualizer
    "load_collected_data",
    "sample_and_visualize",
    "verify_keypoint_mapping",
    # Utils
    "parse_frame_info",
    "find_image_in_subfolders",
    "get_keypoint_names_from_columns",
]
