"""
Dataset download and extraction utilities.
"""

import os
import zipfile
from pathlib import Path
from typing import Optional

import requests


def download_dataset(url: str, output_path: Path) -> Path:
    """
    Download a dataset zip file from Roboflow.

    Args:
        url: Roboflow dataset download URL
        output_path: Path where the zip file will be saved

    Returns:
        Path to the downloaded zip file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"✅ Download complete: {output_path}")
    return output_path


def extract_dataset(zip_path: Path, extract_dir: Path) -> Path:
    """
    Extract a zip file to the specified directory.

    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract files to

    Returns:
        Path to the extraction directory
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"✅ Extraction complete")
    print(f"Extracted files: {os.listdir(extract_dir)}")
    return extract_dir


def download_and_extract(
    url: str,
    output_dir: Path,
    zip_filename: str = "dataset.zip",
    keep_zip: bool = False,
) -> Path:
    """
    Download and extract a Roboflow dataset in one step.

    Args:
        url: Roboflow dataset download URL
        output_dir: Directory to store the extracted dataset
        zip_filename: Name for the downloaded zip file
        keep_zip: Whether to keep the zip file after extraction

    Returns:
        Path to the extraction directory containing train/valid/test folders
    """
    output_dir = Path(output_dir)
    zip_path = output_dir / zip_filename

    # Download
    download_dataset(url, zip_path)

    # Extract
    extract_dataset(zip_path, output_dir)

    # Optionally remove zip
    if not keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"🗑️ Removed zip file: {zip_path}")

    return output_dir
