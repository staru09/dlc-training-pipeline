"""
Video creation from image sequences.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .config import DEFAULT_CONFIG, Config
from .utils import parse_frame_info


def group_frames_by_video(
    base_dir: Path,
    subfolders: Optional[List[str]] = None,
    config: Optional[Config] = None,
) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Group image files by their source video name.

    Args:
        base_dir: Base directory containing image subfolders
        subfolders: List of subfolder names to search
        config: Configuration object

    Returns:
        Dictionary mapping video names to list of (frame_number, image_path) tuples
    """
    if config is None:
        config = DEFAULT_CONFIG
    if subfolders is None:
        subfolders = config.subfolders

    base_dir = Path(base_dir)
    video_groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)

    for sub in subfolders:
        subfolder_path = base_dir / sub
        if not subfolder_path.exists():
            continue

        for img_path in subfolder_path.glob("*.jpg"):
            frame_info = parse_frame_info(img_path.name, config.frame_pattern)
            if frame_info:
                video_name, frame_number = frame_info
                video_groups[video_name].append((frame_number, img_path))

    return dict(video_groups)


def create_video_from_frames(
    frame_list: List[Tuple[int, Path]],
    output_path: Path,
    fps: int = 24,
    codec: str = "mp4v",
) -> Path:
    """
    Create a video from a list of image frames.

    Args:
        frame_list: List of (frame_number, image_path) tuples
        output_path: Path for the output video file
        fps: Frames per second for the video
        codec: FourCC codec code

    Returns:
        Path to the created video
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort frames by frame number
    sorted_frames = sorted(frame_list, key=lambda x: x[0])

    # Get video dimensions from first frame
    sample_img = cv2.imread(str(sorted_frames[0][1]))
    if sample_img is None:
        raise ValueError(f"Could not read first frame: {sorted_frames[0][1]}")

    height, width = sample_img.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for _, img_path in sorted_frames:
            img = cv2.imread(str(img_path))
            if img is not None:
                out.write(img)
    finally:
        out.release()

    return output_path


def create_all_videos(
    base_dir: Path,
    output_dir: Optional[Path] = None,
    subfolders: Optional[List[str]] = None,
    config: Optional[Config] = None,
) -> List[Path]:
    """
    Create videos for all image groups in the dataset.

    Args:
        base_dir: Base directory containing image subfolders
        output_dir: Directory for output videos. Defaults to base_dir/videos
        subfolders: List of subfolder names to search
        config: Configuration object

    Returns:
        List of paths to created videos
    """
    if config is None:
        config = DEFAULT_CONFIG

    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir / "videos"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group frames by video
    video_groups = group_frames_by_video(base_dir, subfolders, config)

    created_videos = []
    for video_name, frame_list in video_groups.items():
        video_path = output_dir / f"{video_name}.mp4"
        create_video_from_frames(
            frame_list, video_path, fps=config.video_fps, codec=config.video_codec
        )
        print(f"✅ Created video: {video_path}")
        created_videos.append(video_path)

    print(f"\n📹 Created {len(created_videos)} video(s) in {output_dir}")
    return created_videos
