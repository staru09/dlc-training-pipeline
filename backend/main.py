from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import (
    DETECTORS,
    POSE_MODELS,
    SUPERANIMAL_DATASETS,
    ensure_dirs,
    settings,
)
from inference import run_gcs_inference, run_inference
from job_manager import create_job, get_job, run_job_in_background
from schemas import (
    GCSInferenceResponse,
    HealthResponse,
    InferenceParams,
    InferenceResponse,
    JobStatusResponse,
    ModelsResponse,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DLC SuperAnimal Inference API",
    description=(
        "Upload a video, pick a SuperAnimal model + detector, "
        "and get back an annotated video with pose data."
    ),
    version="0.2.0",
)


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()
    logger.info("Upload  dir → %s", settings.UPLOAD_DIR.resolve())
    logger.info("Output  dir → %s", settings.OUTPUT_DIR.resolve())


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/", response_model=HealthResponse)
def health_check():
    """Quick liveness probe."""
    return HealthResponse()


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """Return every model / detector / dataset the user can choose from."""
    return ModelsResponse(
        pose_models=POSE_MODELS,
        detectors=DETECTORS,
        superanimal_datasets=SUPERANIMAL_DATASETS,
        defaults={
            "model_name": settings.DEFAULT_MODEL,
            "detector_name": settings.DEFAULT_DETECTOR,
            "superanimal_name": settings.DEFAULT_SUPERANIMAL,
        },
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    video: UploadFile = File(..., description="Video file to run inference on"),
    superanimal_name: str = Form(default=settings.DEFAULT_SUPERANIMAL),
    model_name: str = Form(default=settings.DEFAULT_MODEL),
    detector_name: str = Form(default=settings.DEFAULT_DETECTOR),
    max_individuals: int = Form(default=settings.DEFAULT_MAX_INDIVIDUALS),
    pcutoff: float = Form(default=settings.DEFAULT_PCUTOFF),
    batch_size: int = Form(default=settings.DEFAULT_BATCH_SIZE),
    detector_batch_size: int = Form(default=settings.DEFAULT_DETECTOR_BATCH_SIZE),
    device: str = Form(default=settings.DEFAULT_DEVICE),
):
    """
    Upload a video and start SuperAnimal inference in the background.

    Returns a ``job_id`` immediately. Poll ``GET /jobs/{job_id}`` to
    monitor progress and retrieve results when complete.
    """
    # ── Validate choices ─────────────────────────────────────────────────
    if model_name not in POSE_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_name}'. Choose from: {POSE_MODELS}",
        )
    if detector_name not in DETECTORS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown detector '{detector_name}'. Choose from: {DETECTORS}",
        )
    if superanimal_name not in SUPERANIMAL_DATASETS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown dataset '{superanimal_name}'. "
                f"Choose from: {SUPERANIMAL_DATASETS}"
            ),
        )

    # ── Save upload ──────────────────────────────────────────────────────
    unique_name = f"{uuid.uuid4().hex[:10]}_{video.filename}"
    upload_path = settings.UPLOAD_DIR.resolve() / unique_name

    try:
        contents = await video.read()
        upload_path.write_bytes(contents)
        logger.info("Saved upload → %s (%.1f MB)", upload_path, len(contents) / 1e6)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    # ── Build params ─────────────────────────────────────────────────────
    params = InferenceParams(
        superanimal_name=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        max_individuals=max_individuals,
        pcutoff=pcutoff,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        device=device,
    )

    # ── Create job & launch in background ────────────────────────────────
    job = create_job(
        video_name=video.filename,
        model_name=model_name,
        superanimal_name=superanimal_name,
    )
    job.add_log(f"Video uploaded: {video.filename} ({len(contents) / 1e6:.1f} MB)")

    run_job_in_background(
        job=job,
        run_fn=run_inference,
        video_path=upload_path,
        params=params,
    )

    return InferenceResponse(
        message="Inference job started — poll /jobs/{job_id} for status",
        job_id=job.id,
        video_name=video.filename,
        model_used=model_name,
        detector_used=detector_name,
        superanimal=superanimal_name,
        result_files=[],
    )


# ── GCS-to-GCS inference endpoint ────────────────────────────────────────────


@app.post("/infer/gcs", response_model=GCSInferenceResponse)
async def infer_gcs(
    gcs_input_path: str = Form(
        ...,
        description="GCS input path as bucket_name/folder/UUID.mp4",
    ),
    gcs_output_path: str = Form(
        ...,
        description="GCS output path as bucket_name/folder",
    ),
    superanimal_name: str = Form(default=settings.DEFAULT_SUPERANIMAL),
    model_name: str = Form(default=settings.DEFAULT_MODEL),
    detector_name: str = Form(default=settings.DEFAULT_DETECTOR),
    max_individuals: int = Form(default=settings.DEFAULT_MAX_INDIVIDUALS),
    pcutoff: float = Form(default=settings.DEFAULT_PCUTOFF),
    batch_size: int = Form(default=settings.DEFAULT_BATCH_SIZE),
    detector_batch_size: int = Form(default=settings.DEFAULT_DETECTOR_BATCH_SIZE),
    device: str = Form(default=settings.DEFAULT_DEVICE),
):
    """
    Process a video from GCS. Downloads the input from a GCS bucket, runs
    DLC SuperAnimal inference, uploads results to GCS, and cleans up
    all local files.

    Returns a ``task_id`` immediately. Poll ``GET /jobs/{task_id}`` to
    monitor progress and retrieve results when complete.
    """
    # ── Validate model choices ───────────────────────────────────────────
    if model_name not in POSE_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_name}'. Choose from: {POSE_MODELS}",
        )
    if detector_name not in DETECTORS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown detector '{detector_name}'. Choose from: {DETECTORS}",
        )
    if superanimal_name not in SUPERANIMAL_DATASETS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown dataset '{superanimal_name}'. Choose from: {SUPERANIMAL_DATASETS}",
        )

    # ── Build params ─────────────────────────────────────────────────────
    params = InferenceParams(
        superanimal_name=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        max_individuals=max_individuals,
        pcutoff=pcutoff,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        device=device,
    )

    # ── Extract video name from input path ──────────────────────────────
    video_name = gcs_input_path.rsplit("/", 1)[-1]

    # ── Create job & launch in background ────────────────────────────────
    job = create_job(
        video_name=video_name,
        model_name=model_name,
        superanimal_name=superanimal_name,
    )
    job.add_log(f"GCS task queued: {video_name} from {gcs_input_path}")

    run_job_in_background(
        job=job,
        run_fn=run_gcs_inference,
        gcs_input_path=gcs_input_path,
        gcs_output_path=gcs_output_path,
        params=params,
    )

    return GCSInferenceResponse(
        success=True,
        message="Task queued successfully",
        task_id=job.id,
    )


# ── Job status polling endpoint ──────────────────────────────────────────────


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    """
    Poll this endpoint to monitor a running inference job.

    Returns the current status, elapsed time, captured log lines,
    and (when complete) the list of downloadable result files.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    data = job.to_dict()
    return JobStatusResponse(**data)


# ── Download results ─────────────────────────────────────────────────────────


@app.get("/results/{filename}")
def download_result(filename: str):
    """Download a result file produced by a previous inference run."""
    file_path = settings.OUTPUT_DIR.resolve() / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Determine media type
    media_types = {
        ".mp4": "video/mp4",
        ".h5": "application/x-hdf5",
        ".json": "application/json",
        ".csv": "text/csv",
        ".pickle": "application/octet-stream",
    }
    media_type = media_types.get(file_path.suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )
