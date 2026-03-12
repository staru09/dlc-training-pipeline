from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from config import settings
from schemas import InferenceParams

logger = logging.getLogger(__name__)

_MP4_EXTENSIONS = {".mp4", ".m4v", ".mov"}


def _ensure_mp4(video_path: Path, _report=None) -> Path:
    """Convert non-MP4 videos (e.g. webm/av1) to MP4 H.264 so OpenCV can read them."""
    if video_path.suffix.lower() in _MP4_EXTENSIONS:
        return video_path

    _report = _report or (lambda *a, **kw: None)
    mp4_path = video_path.with_suffix(".mp4")

    logger.info("Converting %s → %s (original codec may not be supported by OpenCV)", video_path.name, mp4_path.name)
    _report(0.03, f"Converting {video_path.suffix} to MP4")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p",
        "-an", str(mp4_path),
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        logger.info(
            "Converted %s → %s (%.2f MB)",
            video_path.name, mp4_path.name, mp4_path.stat().st_size / 1e6,
        )
        return mp4_path
    except subprocess.CalledProcessError as e:
        logger.error(
            "ffmpeg conversion failed for %s (exit code %d)\nstderr: %s",
            video_path.name, e.returncode, e.stderr,
        )
        raise RuntimeError(f"Failed to convert {video_path.name} to MP4: {e.stderr}") from e
    except FileNotFoundError:
        logger.error("ffmpeg binary not found in PATH")
        raise RuntimeError("ffmpeg is not installed")


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

    logger.info("Discovering result files for stem=%s in %s", stem, output_dir)

    for pattern in (
        f"{stem}*.mp4",   # annotated / labeled video
        f"{stem}*.h5",    # pose data (HDF5)
        f"{stem}*.json",  # pose data (JSON)
        f"{stem}*.csv",   # pose data (CSV)
        f"{stem}*.pickle",
    ):
        found = list(output_dir.glob(pattern))
        if found:
            logger.info("  pattern %s → %d file(s): %s", pattern, len(found), [f.name for f in found])
        results.extend(found)

    # Exclude the original upload if it somehow ended up here
    results = [r for r in results if r != video_path]
    results = sorted(set(results))
    logger.info("Discovered %d result file(s) total", len(results))
    return results





def _copy_results_to_output(
    result_files: list[Path],
    output_dir: Path,
    gcs_output_path: str | None = None,
) -> list[str]:
    """Copy result files into the shared output directory and return filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames: list[str] = []
    final_paths: list[Path] = []

    for src in result_files:
        dst = output_dir / src.name
        if src.parent.resolve() != output_dir.resolve():
            logger.info("Copying %s (%.2f MB) → %s", src.name, src.stat().st_size / 1e6, dst)
            shutil.copy2(src, dst)
        
        if dst.suffix.lower() == ".mp4":
            tmp_dst = dst.with_name(f"temp_{dst.name}")
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", str(dst),
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an", str(tmp_dst),
            ]
            logger.info(
                "Re-encoding %s (%.2f MB) → H.264",
                dst.name, dst.stat().st_size / 1e6,
            )
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                tmp_dst.replace(dst)
                logger.info(
                    "Re-encoded %s → %.2f MB",
                    dst.name, dst.stat().st_size / 1e6,
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    "ffmpeg failed for %s (exit code %d)\nstdout: %s\nstderr: %s",
                    dst.name, e.returncode, e.stdout, e.stderr,
                )
                raise RuntimeError(
                    f"ffmpeg re-encode failed for {dst.name} "
                    f"(exit code {e.returncode}): {e.stderr}"
                ) from e
            except FileNotFoundError:
                logger.error("ffmpeg binary not found in PATH")
                raise RuntimeError("ffmpeg is not installed")

        filenames.append(dst.name)
        final_paths.append(dst)

    if gcs_output_path:
        from gcs_utils import upload_to_gcs
        # Upload the processed destination files, not the sources
        upload_to_gcs(final_paths, gcs_output_path)

    return filenames


def run_inference(
    video_path: Path,
    params: InferenceParams,
    gcs_output_path: str | None = None,
    progress_callback=None,
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
    _report = progress_callback or (lambda *a, **kw: None)

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

    _report(0.05, f"Starting DLC inference on {video_path.name}")

    try:
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
    except TypeError as exc:
        # DLC's create_video uses DataFrame.groupby(axis=1) which was removed
        # in pandas 2.x. Pose data (H5/pickle) is already saved by this point,
        # so we log the warning and continue to collect the results.
        if "axis" in str(exc):
            logger.warning(
                "Labeled video creation failed (pandas compat issue): %s. "
                "Pose data (H5/pickle) was saved successfully.", exc,
            )
        else:
            raise

    _report(0.55, "DLC inference complete, discovering output files")
    logger.info("DLC inference finished, searching for output files")

    # Discover outputs — DLC may write next to the video OR into dest_folder
    result_files: list[Path] = []
    for search_dir in {video_path.parent, settings.OUTPUT_DIR.resolve()}:
        result_files.extend(_discover_result_files(video_path, search_dir))

    # Remove corrupt labeled MP4s (DLC's create_video fails on pandas 2.x,
    # leaving tiny empty files).  Files < 1 KB are almost certainly broken.
    corrupt = [f for f in result_files if f.suffix == ".mp4" and f.stat().st_size < 1024]
    if corrupt:
        logger.warning("Removing %d corrupt MP4(s) (< 1KB): %s", len(corrupt), [f.name for f in corrupt])
    result_files = [f for f in result_files if f not in corrupt]

    logger.info(
        "Result files after filtering: %s",
        [(f.name, f"{f.stat().st_size / 1e6:.2f} MB") for f in result_files],
    )

    # ── Create annotated video with our own annotator ────────────────────
    json_files = [f for f in result_files if f.suffix == ".json"]
    if json_files:
        from annotator import create_annotated_video

        _report(0.60, "Creating annotated video")

        json_path = json_files[0]
        annotated_path = settings.OUTPUT_DIR.resolve() / f"{video_path.stem}_annotated.mp4"
        logger.info("Creating annotated video from %s → %s", json_path.name, annotated_path.name)
        try:
            create_annotated_video(
                video_path=video_path,
                predictions_json_path=json_path,
                output_path=annotated_path,
                pcutoff=params.pcutoff,
                progress_callback=progress_callback,
            )
            logger.info(
                "Annotated video created: %s (%.2f MB)",
                annotated_path.name, annotated_path.stat().st_size / 1e6,
            )
            result_files.append(annotated_path)
        except Exception as exc:
            logger.error("Custom annotated video creation failed: %s", exc, exc_info=True)
    else:
        logger.warning("No JSON predictions found — skipping annotated video creation")

    _report(0.90, "Copying and re-encoding results")

    # Consolidate everything into OUTPUT_DIR
    logger.info("Copying %d result file(s) to output dir and re-encoding MP4s", len(result_files))
    filenames = _copy_results_to_output(
        result_files, settings.OUTPUT_DIR.resolve(), gcs_output_path=gcs_output_path,
    )

    _report(0.95, "Results ready")

    logger.info("Inference complete — %d result file(s): %s", len(filenames), filenames)
    return {
        "result_files": filenames,
        "output_dir": settings.OUTPUT_DIR.resolve(),
    }


def run_gcs_inference(
    gcs_input_path: str,
    gcs_output_path: str,
    params: InferenceParams,
    progress_callback=None,
) -> dict:
    """
    Download video from GCS → run inference → upload results → cleanup.

    This is the top-level function spawned as a subprocess by the job manager.

    gcs_input_path: bucket_name/folder/UUID.mp4
    gcs_output_path: bucket_name/folder
    """
    import tempfile

    from gcs_utils import cleanup_local_files, download_from_gcs

    _report = progress_callback or (lambda *a, **kw: None)

    # Parse input: bucket_name/folder/UUID.mp4 → bucket + blob_path
    input_parts = gcs_input_path.split("/", 1)
    input_bucket = input_parts[0]
    blob_path = input_parts[1] if len(input_parts) > 1 else ""

    logger.info(
        "GCS inference starting | input=gs://%s/%s output=gs://%s model=%s",
        input_bucket, blob_path, gcs_output_path, params.model_name,
    )

    # Create a temp directory for the whole job
    tmp_dir = Path(tempfile.mkdtemp(prefix="dlc_gcs_"))
    logger.info("Created temp dir: %s", tmp_dir)
    result = None
    try:
        # 1. Download from GCS
        _report(0.02, f"Downloading video from gs://{input_bucket}/{blob_path}")
        local_video = download_from_gcs(input_bucket, blob_path, tmp_dir)
        logger.info("Video downloaded to %s (%.2f MB)", local_video, local_video.stat().st_size / 1e6)

        # 2. Convert to MP4 if not already (e.g. webm/av1 that OpenCV can't decode)
        local_video = _ensure_mp4(local_video, _report)

        # 3. Run DLC inference (progress 0.05 → 0.95 reported inside)
        result = run_inference(
            video_path=local_video,
            params=params,
            gcs_output_path=gcs_output_path,
            progress_callback=progress_callback,
        )

        logger.info("GCS inference pipeline complete — %d result file(s)", len(result.get("result_files", [])))
        return result
    except Exception:
        logger.error("GCS inference pipeline failed", exc_info=True)
        raise
    finally:
        # 3. Cleanup all local temp files
        logger.info("Cleaning up temp dir: %s", tmp_dir)
        cleanup_local_files(tmp_dir)
        # Also cleanup any output files (they've been uploaded to GCS)
        if result:
            for fname in result.get("result_files", []):
                cleanup_local_files(settings.OUTPUT_DIR.resolve() / fname)

