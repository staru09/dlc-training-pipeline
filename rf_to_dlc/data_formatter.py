"""
DLC-compatible data formatting utilities.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import DEFAULT_CONFIG


def create_single_animal_headers(
    scorer: str, bodyparts: List[str]
) -> List[List[str]]:
    """
    Create the 3-row header structure for single-animal DLC projects.

    Args:
        scorer: Name of the annotator/scorer
        bodyparts: List of bodypart/keypoint names

    Returns:
        List of 3 header rows:
            - Row 0: ['scorer', scorer, scorer, ...]
            - Row 1: ['bodyparts', bp1, bp1, bp2, bp2, ...]
            - Row 2: ['coords', 'x', 'y', 'x', 'y', ...]
    """
    num_coords = len(bodyparts) * 2

    header_rows = [
        ["scorer"] + [scorer] * num_coords,
        ["bodyparts"] + [bp for bp in bodyparts for _ in (0, 1)],
        ["coords"] + ["x", "y"] * len(bodyparts),
    ]

    return header_rows


def create_multi_animal_headers(
    scorer: str,
    individuals: List[str],
    bodyparts: List[str],
) -> List[List[str]]:
    """
    Create the 4-row header structure for multi-animal DLC projects.

    Args:
        scorer: Name of the annotator/scorer
        individuals: List of individual names (e.g., ['individual1', 'individual2'])
        bodyparts: List of bodypart/keypoint names

    Returns:
        List of 4 header rows:
            - Row 0: ['scorer', scorer, scorer, ...]
            - Row 1: ['individuals', individual, individual, ...]
            - Row 2: ['bodyparts', bp1, bp1, bp2, bp2, ...]
            - Row 3: ['coords', 'x', 'y', 'x', 'y', ...]
    """
    num_coords = len(bodyparts) * 2
    individual = individuals[0] if individuals else DEFAULT_CONFIG.default_individual

    header_rows = [
        ["scorer"] + [scorer] * num_coords,
        ["individuals"] + [individual] * num_coords,
        ["bodyparts"] + [bp for bp in bodyparts for _ in (0, 1)],
        ["coords"] + ["x", "y"] * len(bodyparts),
    ]

    return header_rows


def format_dataframe(
    df: pd.DataFrame,
    headers: List[List[str]],
) -> pd.DataFrame:
    """
    Combine header rows with annotation data into DLC format.

    Args:
        df: DataFrame with annotation data (first column is image_name)
        headers: List of header rows from create_*_headers functions

    Returns:
        Combined DataFrame with headers prepended
    """
    df_combined = pd.DataFrame(headers + df.values.tolist())
    return df_combined


def remove_duplicates(
    df: pd.DataFrame,
    header_rows: int = 3,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate image entries while preserving headers.

    Args:
        df: Combined DataFrame with headers
        header_rows: Number of header rows (3 for single-animal, 4 for multi-animal)
        keep: Which duplicate to keep ('first', 'last', or False)

    Returns:
        DataFrame with duplicates removed
    """
    headers = df.iloc[:header_rows]
    data = df.iloc[header_rows:]

    # Remove duplicates based on image name (column 0)
    data_clean = data.drop_duplicates(subset=0, keep=keep).reset_index(drop=True)

    # Combine back
    df_unique = pd.concat([headers, data_clean], ignore_index=True)
    return df_unique


def save_collected_data(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save the formatted DataFrame as a CSV file.

    Args:
        df: Formatted DataFrame with headers
        output_path: Path for the output CSV file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, header=False)
    print(f"✅ CollectedData saved to: {output_path}")
    return output_path


def convert_to_hdf5(
    csv_path: Path,
    output_path: Optional[Path] = None,
    header_rows: int = 3,
) -> Path:
    """
    Convert a CollectedData CSV file to HDF5 format.

    Args:
        csv_path: Path to the CollectedData CSV file
        output_path: Path for the output H5 file. Defaults to same path with .h5 extension.
        header_rows: Number of header rows (3 for single-animal, 4 for multi-animal)

    Returns:
        Path to the saved H5 file
    """
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix(".h5")
    else:
        output_path = Path(output_path)

    # Load with proper multi-index headers
    header_indices = list(range(header_rows))
    df = pd.read_csv(csv_path, header=header_indices, index_col=0)

    # Save to HDF5
    df.to_hdf(output_path, key="df_with_missing", mode="w")
    print(f"✅ HDF5 saved to: {output_path}")

    return output_path
