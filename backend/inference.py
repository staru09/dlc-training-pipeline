"""
Core inference service — wraps deeplabcut.video_inference_superanimal.

This module is framework-agnostic on the FastAPI side; it only depends on
DeepLabCut and standard-lib modules.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from config import settings
from schemas import InferenceParams

logger = logging.getLogger(__name__)


def _discover_result_files(
    video_path: Path,
    output_dir: Path,
) -> list[Path]:
    """
    After DLC inference, find all generated artefacts for the given video.

    DLC typically produces files like:
        <video_stem>_<scorer>.<ext>   where ext ∈ {.h5, .json, .mp4, .csv, .pickle}
    """
    stem = video_path.stem
    results: list[Path] = []

    for pattern in (
        f"{stem}*.mp4",   # annotated / labeled video
        f"{stem}*.h5",    # pose data (HDF5)
        f"{stem}*.json",  # pose data (JSON)
        f"{stem}*.csv",   # pose data (CSV)
        f"{stem}*.pickle",
    ):
        results.extend(output_dir.glob(pattern))

    # Exclude the original upload if it somehow ended up here
    results = [r for r in results if r != video_path]
    return sorted(set(results))


def _copy_results_to_output(
    result_files: list[Path],
    output_dir: Path,
) -> list[str]:
    """Copy result files into the shared output directory and return filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames: list[str] = []

    for src in result_files:
        dst = output_dir / src.name
        if src.parent.resolve() != output_dir.resolve():
            shutil.copy2(src, dst)
        filenames.append(src.name)

    return filenames


def run_inference(
    video_path: Path,
    params: InferenceParams,
) -> dict:
    """
    Run SuperAnimal inference on a single video.

    Parameters
    ----------
    video_path : Path
        Absolute path to the uploaded video file.
    params : InferenceParams
        User-selected model configuration.

    Returns
    -------
    dict with keys:
        - result_files : list[str]  — filenames available for download
        - output_dir   : Path       — directory where results live
    """
    # Lazy import so the module loads even without DLC installed (e.g. tests)
    from deeplabcut import video_inference_superanimal

    dest_folder = str(settings.OUTPUT_DIR.resolve())

    logger.info(
        "Starting inference | video=%s model=%s detector=%s dataset=%s device=%s",
        video_path.name,
        params.model_name,
        params.detector_name,
        params.superanimal_name,
        params.device,
    )

    video_inference_superanimal(
        videos=[str(video_path)],
        superanimal_name=params.superanimal_name,
        model_name=params.model_name,
        detector_name=params.detector_name,
        max_individuals=params.max_individuals,
        pcutoff=params.pcutoff,
        batch_size=params.batch_size,
        detector_batch_size=params.detector_batch_size,
        video_adapt=False,
        device=params.device,
        dest_folder=dest_folder,
    )

    # Discover outputs — DLC may write next to the video OR into dest_folder
    result_files: list[Path] = []
    for search_dir in {video_path.parent, settings.OUTPUT_DIR.resolve()}:
        result_files.extend(_discover_result_files(video_path, search_dir))

    # Consolidate everything into OUTPUT_DIR
    filenames = _copy_results_to_output(result_files, settings.OUTPUT_DIR.resolve())

    logger.info("Inference complete — %d result file(s)", len(filenames))
    return {
        "result_files": filenames,
        "output_dir": settings.OUTPUT_DIR.resolve(),
    }
