from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def download_from_gcs(
    bucket_name: str,
    blob_path: str,
    local_dir: Path,
) -> Path:
    """
    Download a file from Google Cloud Storage to a local directory.

    Parameters
    ----------
    bucket_name : str
        GCS bucket name (no ``gs://`` prefix).
    blob_path : str
        Path to the blob inside the bucket (e.g. ``processed_videos/abc.mp4``).
    local_dir : Path
        Local directory to save the downloaded file in.

    Returns
    -------
    Path to the downloaded local file.
    """
    from google.cloud import storage

    local_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(blob_path).name
    local_path = local_dir / filename

    logger.info(
        "Downloading gs://%s/%s → %s", bucket_name, blob_path, local_path,
    )

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        logger.error("Blob not found: gs://%s/%s", bucket_name, blob_path)
        raise FileNotFoundError(f"GCS blob not found: gs://{bucket_name}/{blob_path}")

    blob.reload()
    logger.info("Blob size on GCS: %.2f MB", (blob.size or 0) / 1e6)

    blob.download_to_filename(str(local_path))

    size_mb = local_path.stat().st_size / 1e6
    logger.info("Downloaded %.2f MB → %s", size_mb, local_path)
    return local_path


def upload_to_gcs(
    local_files: list[Path],
    gcs_output_path: str,
) -> list[str]:
    """
    Upload local files to Google Cloud Storage.

    Parameters
    ----------
    local_files : list[Path]
        Local files to upload.
    gcs_output_path : str
        GCS destination in ``bucket/folder`` format
        (e.g. ``dlc_bucket/dlc_output_main``).

    Returns
    -------
    List of ``gs://`` URIs for the uploaded files.
    """
    from google.cloud import storage

    # Strip optional gs:// prefix and parse bucket/prefix
    path = gcs_output_path.replace("gs://", "")
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    uploaded_uris: list[str] = []

    for file_path in local_files:
        blob_name = f"{prefix}{file_path.name}"
        blob = bucket.blob(blob_name)
        size_mb = file_path.stat().st_size / 1e6
        logger.info(
            "Uploading %s (%.2f MB) → gs://%s/%s",
            file_path.name, size_mb, bucket_name, blob_name,
        )
        blob.upload_from_filename(str(file_path))
        uploaded_uris.append(f"gs://{bucket_name}/{blob_name}")
        logger.info("Uploaded %s successfully", file_path.name)

    logger.info("Uploaded %d file(s) to GCS: %s", len(uploaded_uris), uploaded_uris)
    return uploaded_uris


def cleanup_local_files(*paths: Path) -> None:
    """
    Delete local files and directories.  Silently skips paths that
    don't exist.
    """
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
                logger.info("Cleaned up directory: %s", p)
            elif p.is_file():
                p.unlink()
                logger.info("Cleaned up file: %s", p)
        except Exception as exc:
            logger.warning("Cleanup failed for %s: %s", p, exc)
