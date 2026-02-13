#!/usr/bin/env python3
"""
Convert Roboflow keypoint dataset to DeepLabCut format.

Usage:
    python convert_data.py --input dataset --output labeled-data --scorer aru
    python convert_data.py --input dataset --output labeled-data --scorer aru --video-name milo_15s
"""

import json
import shutil
import argparse
from pathlib import Path
import pandas as pd


def load_keypoints_from_coco(annotation_path: Path) -> tuple[list, list]:
    """Load keypoint annotations from COCO JSON format."""
    if not annotation_path.exists():
        return [], []
    
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    # Map image_id -> file_name
    image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}
    
    # Find keypoint names from categories
    keypoint_names = []
    for cat in data.get("categories", []):
        if "keypoints" in cat and cat["keypoints"]:
            keypoint_names = cat["keypoints"]
            break
    
    records = []
    for ann in data["annotations"]:
        if "keypoints" not in ann or len(ann["keypoints"]) == 0:
            continue
        
        image_name = image_id_to_name[ann["image_id"]]
        kpts = ann["keypoints"]
        
        # Extract x, y coordinates (visibility flag 2 = labeled and visible)
        coords = []
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i:i+3]
            # v=2: visible, v=1: occluded, v=0: not labeled
            coords.extend([x, y] if v >= 1 else [None, None])
        
        records.append([image_name] + coords)
    
    return keypoint_names, records


def convert_to_dlc_format(
    input_dir: Path,
    output_dir: Path,
    scorer: str = "aru",
    video_name: str = "milo_15s"
) -> bool:
    """Convert Roboflow dataset to DeepLabCut format."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Scorer: {scorer}")
    
    # Find all annotation files
    subfolders = ['train', 'valid', 'test']
    all_keypoints = []
    all_records = []
    all_images = {}
    
    for sub in subfolders:
        sub_dir = input_dir / sub
        if not sub_dir.exists():
            continue
        
        # Load annotations
        ann_path = sub_dir / "_annotations.coco.json"
        keypoints, records = load_keypoints_from_coco(ann_path)
        
        if keypoints and not all_keypoints:
            all_keypoints = keypoints
        
        # Add video path prefix to image names
        for record in records:
            record[0] = f"labeled-data/{video_name}/{record[0]}"
        all_records.extend(records)
        
        # Collect images
        for img in sub_dir.glob("*.jpg"):
            all_images[img.name] = img
        for img in sub_dir.glob("*.png"):
            all_images[img.name] = img
    
    if not all_keypoints:
        print("ERROR: No keypoints found in annotations")
        return False
    
    print(f"\nFound {len(all_keypoints)} keypoints: {all_keypoints}")
    print(f"Found {len(all_records)} annotations")
    print(f"Found {len(all_images)} images")
    
    # Build DataFrame
    columns = ["image_name"]
    for part in all_keypoints:
        columns += [f"{part}_x", f"{part}_y"]
    
    df = pd.DataFrame(all_records, columns=columns)
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset="image_name", keep="first")
    print(f"After dedup: {len(df)} unique frames")
    
    # Create DLC header rows
    header_rows = [
        ["scorer"] + [scorer] * (len(columns) - 1),
        ["bodyparts"] + [bp for bp in all_keypoints for _ in (0, 1)],
        ["coords"] + ["x", "y"] * len(all_keypoints),
    ]
    
    # Combine headers with data
    df_combined = pd.DataFrame(header_rows + df.values.tolist())
    
    # Save CSV
    csv_path = output_dir / f"CollectedData_{scorer}.csv"
    df_combined.to_csv(csv_path, index=False, header=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    
    # Create H5 file with proper MultiIndex
    df_h5 = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    h5_path = output_dir / f"CollectedData_{scorer}.h5"
    df_h5.to_hdf(h5_path, key="df_with_missing", mode="w")
    print(f"✓ Saved H5: {h5_path}")
    
    # Copy images
    print(f"\nCopying {len(all_images)} images...")
    copied = 0
    for img_name, img_path in all_images.items():
        dest = output_dir / img_name
        if not dest.exists():
            shutil.copy2(img_path, dest)
            copied += 1
    print(f"✓ Copied {copied} images")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Roboflow keypoint dataset to DeepLabCut format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_data.py --input dataset --output labeled-data
  python convert_data.py --input dataset --output my_project/labeled-data --scorer john
  python convert_data.py --input dataset --output labeled-data --video-name my_video
        """
    )
    parser.add_argument("--input", "-i", required=True, help="Roboflow dataset folder (with train/valid/test)")
    parser.add_argument("--output", "-o", required=True, help="Output labeled-data folder")
    parser.add_argument("--scorer", "-s", default="aru", help="Scorer name (default: aru)")
    parser.add_argument("--video-name", "-v", default="milo_15s", help="Video/folder name (default: milo_15s)")
    
    args = parser.parse_args()
    
    convert_to_dlc_format(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        scorer=args.scorer,
        video_name=args.video_name
    )


if __name__ == "__main__":
    main()
