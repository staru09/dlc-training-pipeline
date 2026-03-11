"""
Backend configuration — model registry, paths, and defaults.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with sensible defaults."""

    # Directories (relative to backend/)
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs")
    
    # Cloud Storage
    GCS_INPUT_PATH: str = "datacam_videos/processed_videos"  # bucket/folder for input videos
    GCS_OUTPUT_BUCKET: str | None = None  # e.g. "gs://dlc_bucket/dlc_output_main"

    # Defaults
    DEFAULT_SUPERANIMAL: str = "superanimal_quadruped"
    DEFAULT_MODEL: str = "hrnet_w32"
    DEFAULT_DETECTOR: str = "fasterrcnn_resnet50_fpn_v2"
    DEFAULT_MAX_INDIVIDUALS: int = 1
    DEFAULT_PCUTOFF: float = 0.1
    DEFAULT_BATCH_SIZE: int = 1
    DEFAULT_DETECTOR_BATCH_SIZE: int = 1
    DEFAULT_DEVICE: str = "auto"

    class Config:
        env_prefix = "DLC_"


settings = Settings()

# ── Registries ───────────────────────────────────────────────────────────────

POSE_MODELS: list[str] = [
    "resnet_50",
    "resnet_101",
    "hrnet_w18",
    "hrnet_w32",
    "hrnet_w48",
]

DETECTORS: list[str] = [
    "fasterrcnn_resnet50_fpn_v2",
]

SUPERANIMAL_DATASETS: list[str] = [
    "superanimal_quadruped",
    "superanimal_topviewmouse",
]


def ensure_dirs() -> None:
    """Create upload/output directories if they don't exist."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
