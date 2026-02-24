"""
FastAPI application — endpoints for DeepLabCut SuperAnimal video inference.

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

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
from inference import run_inference
from schemas import (
    HealthResponse,
    InferenceParams,
    InferenceResponse,
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
    version="0.1.0",
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
    Upload a video and run SuperAnimal inference.

    Returns paths to the annotated video and pose-data files,
    which can be downloaded via ``GET /results/{filename}``.
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
    suffix = Path(video.filename).suffix or ".mp4"
    unique_name = f"{uuid.uuid4().hex[:10]}_{video.filename}"
    upload_path = settings.UPLOAD_DIR.resolve() / unique_name

    try:
        contents = await video.read()
        upload_path.write_bytes(contents)
        logger.info("Saved upload → %s (%.1f MB)", upload_path, len(contents) / 1e6)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    # ── Run inference ────────────────────────────────────────────────────
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

    try:
        result = run_inference(video_path=upload_path, params=params)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return InferenceResponse(
        message="Inference complete",
        video_name=video.filename,
        model_used=model_name,
        detector_used=detector_name,
        superanimal=superanimal_name,
        result_files=result["result_files"],
    )


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
