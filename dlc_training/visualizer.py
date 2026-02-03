"""Visualization functions for DLC predictions."""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import pandas as pd
import deeplabcut


def create_labeled_video(
    config_path: Union[str, Path],
    videos: List[str],
    shuffle: int = 1,
    pcutoff: float = 0.6,
    draw_skeleton: bool = True,
    displayedbodyparts: str = "all",
    trailpoints: int = 0,
    codec: str = "mp4v",
    outputframerate: Optional[float] = None,
    destfolder: Optional[str] = None,
) -> None:
    """Create video with pose predictions overlaid.
    
    Args:
        config_path: Path to project config.yaml
        videos: Video paths to label
        shuffle: Training shuffle index
        pcutoff: Confidence threshold for displaying keypoints
        draw_skeleton: Draw skeleton connections
        displayedbodyparts: "all" or list of bodypart names
        trailpoints: Number of trailing points to show
        codec: Video codec (e.g., "mp4v", "avc1")
        outputframerate: Output video framerate
        destfolder: Destination folder for output videos
        
    Example:
        >>> create_labeled_video(
        ...     config_path="./config.yaml",
        ...     videos=["./videos/test.mp4"],
        ...     pcutoff=0.5,
        ...     draw_skeleton=True
        ... )
    """
    kwargs = {
        "config": str(config_path),
        "videos": videos,
        "shuffle": shuffle,
        "pcutoff": pcutoff,
        "draw_skeleton": draw_skeleton,
        "displayedbodyparts": displayedbodyparts,
        "trailpoints": trailpoints,
        "codec": codec,
    }
    
    if outputframerate:
        kwargs["outputframerate"] = outputframerate
    if destfolder:
        kwargs["destfolder"] = destfolder
        
    deeplabcut.create_labeled_video(**kwargs)


def plot_trajectories(
    config_path: Union[str, Path],
    videos: List[str],
    shuffle: int = 1,
    showfigures: bool = False,
) -> None:
    """Plot keypoint trajectories over video frames.
    
    Creates plots showing x/y coordinates and likelihood over time
    for each bodypart.
    
    Args:
        config_path: Path to project config.yaml
        videos: Video paths to plot
        shuffle: Training shuffle index
        showfigures: Display plots interactively
    """
    deeplabcut.plot_trajectories(
        config=str(config_path),
        videos=videos,
        shuffle=shuffle,
        showfigures=showfigures,
    )


def create_labeled_video_manual(
    csv_path: Union[str, Path],
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    confidence_threshold: float = 0.4,
    dot_radius: int = 8,
    font_scale: float = 1.0,
    show_labels: bool = True,
) -> None:
    """Create labeled video using OpenCV (fallback for DLC issues).
    
    Use this when deeplabcut.create_labeled_video() fails.
    
    Args:
        csv_path: Path to DLC predictions CSV
        video_path: Path to source video
        output_path: Path for output labeled video
        confidence_threshold: Minimum confidence to show keypoint
        dot_radius: Radius of keypoint markers
        font_scale: Font scale for labels
        show_labels: Show bodypart names and confidence
        
    Example:
        >>> create_labeled_video_manual(
        ...     csv_path="./video_DLC_predictions.csv",
        ...     video_path="./video.mp4",
        ...     output_path="./video_labeled.mp4"
        ... )
    """
    # Load DLC CSV with multi-level headers
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    df = df.drop(index=0).reset_index(drop=True)
    
    # Get bodyparts
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df):
            break
            
        for bp in bodyparts:
            try:
                part_data = df.loc[frame_idx].xs(bp, level=1, axis=1)
                x = float(part_data["x"].iloc[0])
                y = float(part_data["y"].iloc[0])
                p = float(part_data["likelihood"].iloc[0])
            except Exception:
                continue
                
            if pd.notna(x) and pd.notna(y) and p >= confidence_threshold:
                center = (int(x), int(y))
                cv2.circle(frame, center, dot_radius, (0, 0, 255), -1)
                
                if show_labels:
                    label = f"{bp}: {p:.2f}"
                    text_pos = (center[0] + 10, center[1] - 10)
                    cv2.putText(frame, label, text_pos, font, font_scale, 
                               (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_path}")
