"""
COCO annotation parsing and conversion to DLC format.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG, Config


def load_coco_annotations(annotation_path: Path) -> Dict[str, Any]:
    """
    Load COCO format annotations from a JSON file.

    Args:
        annotation_path: Path to the _annotations.coco.json file

    Returns:
        Dictionary containing the parsed JSON data
    """
    annotation_path = Path(annotation_path)
    with open(annotation_path, "r") as f:
        return json.load(f)


def get_keypoint_names(data: Dict[str, Any], category_index: int = 2) -> List[str]:
    """
    Extract keypoint names from COCO annotation data.

    Args:
        data: Parsed COCO annotation data
        category_index: Index of the category containing keypoints

    Returns:
        List of keypoint names
    """
    categories = data.get("categories", [])
    if category_index < len(categories):
        return categories[category_index].get("keypoints", [])
    return []


def extract_keypoints(
    annotation: Dict[str, Any],
    keypoint_names: List[str],
    visibility_mode: int = 2,
) -> List[Optional[float]]:
    """
    Extract x/y coordinates from a single annotation's keypoints.

    Args:
        annotation: Single annotation dictionary with 'keypoints' field
        keypoint_names: List of keypoint names for reference
        visibility_mode: Minimum visibility flag to include (0, 1, or 2)
            - 0: not labeled
            - 1: labeled but not visible
            - 2: labeled and visible

    Returns:
        List of [x1, y1, x2, y2, ...] with None for invisible/unlabeled points
    """
    kpts = annotation.get("keypoints", [])
    if not kpts:
        return [None] * (len(keypoint_names) * 2)

    coords = []
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i : i + 3]
        if v >= visibility_mode:
            coords.extend([x, y])
        else:
            coords.extend([None, None])

    return coords


def parse_all_annotations(
    base_dir: Path,
    subfolders: Optional[List[str]] = None,
    image_prefix: str = "labeled-data/video",
    config: Optional[Config] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse all COCO annotations from dataset subfolders.

    Args:
        base_dir: Base directory containing train/valid/test folders
        subfolders: List of subfolder names to search
        image_prefix: Prefix to add to image paths in the output
        config: Configuration object
        verbose: Show progress bars

    Returns:
        Tuple of (DataFrame with annotations, list of keypoint names)
    """
    from tqdm import tqdm

    if config is None:
        config = DEFAULT_CONFIG
    if subfolders is None:
        subfolders = config.subfolders

    base_dir = Path(base_dir)
    records = []
    keypoint_names = []

    # First pass: count total annotations for progress bar
    total_annotations = 0
    annotation_files = []
    for sub in subfolders:
        ann_path = base_dir / sub / config.annotation_filename
        if ann_path.exists():
            annotation_files.append((sub, ann_path))
            data = load_coco_annotations(ann_path)
            total_annotations += len(data.get("annotations", []))

    if verbose:
        print(f"📂 Found {len(annotation_files)} annotation file(s) with {total_annotations} total annotations")

    # Second pass: parse with progress
    pbar = tqdm(total=total_annotations, desc="Parsing annotations", disable=not verbose)
    skipped_count = 0

    for sub, ann_path in annotation_files:
        if verbose:
            tqdm.write(f"📄 Processing {sub}/_annotations.coco.json...")

        data = load_coco_annotations(ann_path)

        # Map image_id -> file_name
        image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}

        # Get keypoint names (only need to do this once)
        if not keypoint_names:
            keypoint_names = get_keypoint_names(data)
            if verbose and keypoint_names:
                tqdm.write(f"🔑 Found {len(keypoint_names)} keypoints: {', '.join(keypoint_names[:5])}{'...' if len(keypoint_names) > 5 else ''}")

        # Extract keypoints from each annotation
        for ann in data["annotations"]:
            pbar.update(1)

            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                skipped_count += 1
                continue

            image_name = image_id_to_name.get(ann["image_id"], "unknown")
            full_image_path = f"{image_prefix}/{image_name}"

            coords = extract_keypoints(
                ann, keypoint_names, visibility_mode=config.visibility_mode
            )
            records.append([full_image_path] + coords)

    pbar.close()

    if verbose:
        print(f"✅ Parsed {len(records)} annotations")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} annotations without keypoints")

    # Build DataFrame
    columns = ["image_name"]
    for part in keypoint_names:
        columns.extend([f"{part}_x", f"{part}_y"])

    df = pd.DataFrame(records, columns=columns)
    return df, keypoint_names


def find_missing_keypoints(
    base_dir: Path,
    subfolders: Optional[List[str]] = None,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Find annotations that are missing keypoints.

    Args:
        base_dir: Base directory containing train/valid/test folders
        subfolders: List of subfolder names to search
        config: Configuration object

    Returns:
        DataFrame with columns [annotation_id, image_id, file_name]
    """
    if config is None:
        config = DEFAULT_CONFIG
    if subfolders is None:
        subfolders = config.subfolders

    base_dir = Path(base_dir)
    skipped = []

    for sub in subfolders:
        ann_path = base_dir / sub / config.annotation_filename
        if not ann_path.exists():
            continue

        data = load_coco_annotations(ann_path)
        image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}

        for ann in data["annotations"]:
            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                skipped.append(
                    {
                        "annotation_id": ann.get("id"),
                        "image_id": ann.get("image_id"),
                        "file_name": image_id_to_name.get(ann["image_id"], "UNKNOWN"),
                    }
                )

    return pd.DataFrame(skipped)
