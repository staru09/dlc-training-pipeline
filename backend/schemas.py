from __future__ import annotations

from pydantic import BaseModel, Field

from config import (
    DETECTORS,
    POSE_MODELS,
    SUPERANIMAL_DATASETS,
    settings,
)


class InferenceParams(BaseModel):
    """Parameters the user can tweak per inference request."""

    superanimal_name: str = Field(
        default=settings.DEFAULT_SUPERANIMAL,
        description="SuperAnimal dataset name",
        json_schema_extra={"enum": SUPERANIMAL_DATASETS},
    )
    model_name: str = Field(
        default=settings.DEFAULT_MODEL,
        description="Pose-estimation architecture",
        json_schema_extra={"enum": POSE_MODELS},
    )
    detector_name: str = Field(
        default=settings.DEFAULT_DETECTOR,
        description="Object detector (required for PyTorch top-down)",
        json_schema_extra={"enum": DETECTORS},
    )
    max_individuals: int = Field(
        default=settings.DEFAULT_MAX_INDIVIDUALS,
        ge=1,
        le=100,
        description="Max animals in the frame",
    )
    pcutoff: float = Field(
        default=settings.DEFAULT_PCUTOFF,
        ge=0.0,
        le=1.0,
        description="Confidence cutoff for predictions",
    )
    batch_size: int = Field(
        default=settings.DEFAULT_BATCH_SIZE,
        ge=1,
        description="Batch size for pose model",
    )
    detector_batch_size: int = Field(
        default=settings.DEFAULT_DETECTOR_BATCH_SIZE,
        ge=1,
        description="Batch size for detector",
    )
    device: str = Field(
        default=settings.DEFAULT_DEVICE,
        description="Device for inference: 'auto', 'cuda', 'cuda:0', 'cpu'",
    )


class InferenceResponse(BaseModel):
    """Returned after an inference job is submitted."""

    message: str
    job_id: str = Field(description="Job ID — poll /jobs/{job_id} for status")
    video_name: str
    model_used: str
    detector_used: str
    superanimal: str
    result_files: list[str] = Field(
        default_factory=list,
        description="Filenames available for download at /results/{filename}",
    )


class ModelsResponse(BaseModel):
    """Lists every option the user can pick."""

    pose_models: list[str]
    detectors: list[str]
    superanimal_datasets: list[str]
    defaults: dict[str, str]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "dlc-inference-api"


class JobStatusResponse(BaseModel):
    """Returned by the job polling endpoint."""

    job_id: str
    status: str = Field(description="queued | running | completed | failed")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress from 0.0 to 1.0",
    )
    video_name: str = ""
    model_name: str = ""
    superanimal_name: str = ""
    elapsed_seconds: float | None = None
    logs: list[dict] = Field(default_factory=list)
    result_files: list[str] = Field(default_factory=list)
    error: str | None = None


class GCSInferenceResponse(BaseModel):
    """Returned after a GCS inference task is submitted."""

    success: bool = True
    message: str
    task_id: str = Field(description="Job ID — poll /jobs/{task_id} for status")


