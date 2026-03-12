"""
Custom video annotator — draws keypoints + bounding boxes onto the original
video using OpenCV, bypassing DLC's broken create_video (pandas 2.x issue).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Colour palette (one per keypoint index, cycles if >len) ──────────────────
_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
    (200, 100, 50), (50, 200, 100), (100, 50, 200), (220, 220, 0),
    (0, 220, 220), (220, 0, 220), (128, 128, 0), (0, 128, 128),
]
_BBOX_COLOR = (0, 255, 0)  # green


def create_annotated_video(
    video_path: Path,
    predictions_json_path: Path,
    output_path: Path | None = None,
    pcutoff: float = 0.1,
    dot_size: int = 4,
    draw_bboxes: bool = True,
    progress_callback=None,
) -> Path:
    """
    Overlay keypoints (and optionally bboxes) from the DLC JSON predictions
    onto the original video.

    Parameters
    ----------
    video_path : Path
        Original input video.
    predictions_json_path : Path
        The ``*_before_adapt.json`` or ``*.json`` file produced by DLC.
    output_path : Path | None
        Where to save the annotated video.  Defaults to
        ``<video_stem>_annotated.mp4`` next to the predictions file.
    pcutoff : float
        Only draw keypoints with confidence ≥ pcutoff.
    dot_size : int
        Radius of keypoint circles.
    draw_bboxes : bool
        Whether to draw detection bounding boxes.

    Returns
    -------
    Path to the created annotated video.
    """
    # ── Load predictions ─────────────────────────────────────────────────
    with open(predictions_json_path, "r") as f:
        predictions: list[dict] = json.load(f)

    # ── Open source video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        output_path = predictions_json_path.parent / f"{video_path.stem}_annotated.mp4"

    # Use mp4v codec (universally supported)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logger.info(
        "Annotating %s → %s  (%d frames, %.1f fps, %dx%d)",
        video_path.name, output_path.name, total_frames, fps, width, height,
    )

    # Progress range for annotation phase: 0.60 → 0.90
    PROG_START, PROG_END = 0.60, 0.90

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(predictions):
            pred = predictions[frame_idx]
            _draw_frame(frame, pred, pcutoff, dot_size, draw_bboxes)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 200 == 0:
            logger.info("  frame %d / %d", frame_idx, total_frames)

        # Report frame-level progress
        if progress_callback and total_frames > 0 and frame_idx % 50 == 0:
            frac = frame_idx / total_frames
            prog = PROG_START + frac * (PROG_END - PROG_START)
            progress_callback(prog, f"Annotating: frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    logger.info("Annotated video saved → %s", output_path)
    return output_path


def _draw_frame(
    frame: np.ndarray,
    pred: dict,
    pcutoff: float,
    dot_size: int,
    draw_bboxes: bool,
) -> None:
    """Draw keypoints and bboxes for one frame (in-place)."""

    individuals = pred.get("bodyparts", [])
    bboxes = pred.get("bboxes", [])
    bbox_scores = pred.get("bbox_scores", [])

    for ind_idx, keypoints in enumerate(individuals):
        # keypoints is a list of [x, y, confidence]
        for kp_idx, kp in enumerate(keypoints):
            x, y, conf = kp[0], kp[1], kp[2]
            if conf < pcutoff:
                continue
            colour = _PALETTE[kp_idx % len(_PALETTE)]
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(frame, (cx, cy), dot_size, colour, -1, cv2.LINE_AA)

    if draw_bboxes:
        for i, bbox in enumerate(bboxes):
            # bbox is [x, y, w, h]
            bx, by, bw, bh = bbox
            x1, y1 = int(round(bx)), int(round(by))
            x2, y2 = int(round(bx + bw)), int(round(by + bh))
            score = bbox_scores[i] if i < len(bbox_scores) else 0.0
            cv2.rectangle(frame, (x1, y1), (x2, y2), _BBOX_COLOR, 2, cv2.LINE_AA)
            label = f"{score:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BBOX_COLOR, 1, cv2.LINE_AA,
            )
