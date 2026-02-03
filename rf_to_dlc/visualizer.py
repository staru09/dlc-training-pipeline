"""
Annotation visualization and testing utilities.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from .config import DEFAULT_CONFIG
from .utils import find_image_in_subfolders, get_keypoint_names_from_columns


def load_collected_data(
    csv_path: Path,
    header_rows: int = 3,
) -> pd.DataFrame:
    """
    Load and parse a CollectedData CSV file.

    Args:
        csv_path: Path to the CollectedData CSV file
        header_rows: Number of header rows (3 for single-animal, 4 for multi-animal)

    Returns:
        DataFrame with proper column names extracted from headers
    """
    csv_path = Path(csv_path)
    df_raw = pd.read_csv(csv_path, header=None)

    # Extract bodypart names and coordinate labels
    bodyparts = df_raw.iloc[header_rows - 2, 1:].tolist()
    coords = df_raw.iloc[header_rows - 1, 1:].tolist()

    # Reconstruct column names
    column_names = ["image_name"] + [
        f"{bp}_{coord}" for bp, coord in zip(bodyparts, coords)
    ]

    # Extract data starting after headers
    df_clean = df_raw.iloc[header_rows:].copy()
    df_clean.columns = column_names
    df_clean["image_name"] = df_clean["image_name"].apply(lambda x: Path(x).name)
    df_clean = df_clean.reset_index(drop=True)

    return df_clean


def sample_and_visualize(
    df: pd.DataFrame,
    base_dir: Path,
    sample_size: int = 5,
    subfolders: Optional[List[str]] = None,
    figsize: tuple = (8, 8),
    save_dir: Optional[Path] = None,
) -> None:
    """
    Sample images and visualize their annotated keypoints.

    Args:
        df: DataFrame with annotation data (from load_collected_data)
        base_dir: Base directory containing image subfolders
        sample_size: Number of images to sample
        subfolders: List of subfolder names to search
        figsize: Figure size for plots
        save_dir: Optional directory to save visualizations
    """
    if subfolders is None:
        subfolders = DEFAULT_CONFIG.subfolders

    base_dir = Path(base_dir)

    # Extract keypoint names
    keypoints = get_keypoint_names_from_columns(df.columns.tolist())

    # Sample images
    samples = df.sample(min(sample_size, len(df)))

    for _, row in samples.iterrows():
        image_file = row["image_name"]
        image_path = find_image_in_subfolders(image_file, base_dir, subfolders)

        if image_path is None:
            print(f"⚠️ Image not found: {image_file}")
            continue

        # Load and display image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=figsize)
        plt.imshow(image_rgb)
        plt.title(image_file)

        # Plot keypoints
        for kp in keypoints:
            x = row.get(f"{kp}_x")
            y = row.get(f"{kp}_y")
            if pd.notna(x) and pd.notna(y):
                plt.plot(float(x), float(y), "ro", markersize=8)
                plt.text(float(x) + 3, float(y) + 3, kp, color="yellow", fontsize=8)

        plt.axis("off")

        if save_dir:
            save_path = Path(save_dir) / f"viz_{image_file}"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            print(f"💾 Saved: {save_path}")

        plt.show()


def verify_keypoint_mapping(
    csv_path: Path,
    image_dir: Path,
    num_samples: int = 5,
    header_rows: int = 3,
    subfolders: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Convenience function to verify keypoint annotations are mapped correctly.

    Args:
        csv_path: Path to the CollectedData CSV file
        image_dir: Base directory containing image subfolders
        num_samples: Number of images to verify
        header_rows: Number of header rows in CSV
        subfolders: List of subfolder names to search
        save_dir: Optional directory to save visualizations
    """
    print(f"📊 Loading annotations from: {csv_path}")
    df = load_collected_data(csv_path, header_rows)

    print(f"📸 Sampling {num_samples} image(s) for verification...")
    sample_and_visualize(
        df,
        image_dir,
        sample_size=num_samples,
        subfolders=subfolders,
        save_dir=save_dir,
    )

    print("✅ Verification complete!")
