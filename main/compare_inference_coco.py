
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION / MAPPING
# ============================================================================

# SuperAnimal → Roboflow mapping
SUPERANIMAL_TO_ROBOFLOW = {
    "nose": "nose",
    "right_eye": "right_eye",
    "left_eye": "left_eye",
    "right_earbase": "right_ear_base",
    "left_earbase": "left_ear_base",
    "throat_end": "neck",
    "neck_base": "whithers",
    "back_base": "spine_1",
    "back_end": "spine_2",
    "tail_end": "tail_tip",
    "front_right_knee": "right_front_wrist",
    "front_right_paw": "right_front_paw",
    "front_left_knee": "left_front_wrist",
    "front_left_paw": "left_front_paw",
    "back_right_knee": "right_back_wrist",
    "back_right_paw": "right_back_paw",
    "back_left_knee": "left_back_wrist",
    "back_left_paw": "left_back_paw",
}

# The list of keypoints we want to compare (values in the mapping above)
TARGET_KEYPOINTS = list(SUPERANIMAL_TO_ROBOFLOW.values())

def load_coco_annotations(json_path: str) -> Dict[str, Dict]:
    """
    Loads COCO annotations and returns a dict mapping:
    image_filename -> {keypoint_name: (x, y, v)}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Map category ID to keypoint names
    # Assuming standard COCO format where categories list keypoints
    # We look for the category that has keypoints
    kp_names = []
    cat_id = None
    for cat in data['categories']:
        if 'keypoints' in cat:
            kp_names = cat['keypoints']
            cat_id = cat['id']
            break
            
    if not kp_names:
        raise ValueError("Could not find keypoint names in COCO categories.")

    # 2. Map image ID to filename
    img_id_to_name = {img['id']: img['file_name'] for img in data['images']}

    # 3. Extract annotations
    annotations = {}
    
    for ann in data['annotations']:
        if ann['category_id'] != cat_id:
            continue
            
        img_name = img_id_to_name.get(ann['image_id'])
        if not img_name:
            continue

        raw_kps = ann.get('keypoints', [])
        # COCO keypoints: [x, y, v, x, y, v, ...]
        
        kp_dict = {}
        for i, kp_name in enumerate(kp_names):
            base = i * 3
            if base + 2 < len(raw_kps):
                x, y, v = raw_kps[base], raw_kps[base+1], raw_kps[base+2]
                kp_dict[kp_name] = (x, y, v)
        
        annotations[img_name] = kp_dict

    return annotations

def load_dlc_predictions(csv_path: str) -> pd.DataFrame:
    """
    Loads DLC inference CSV. 
    Expects MultiIndex header (scorer, bodyparts, coords).
    Returns a DataFrame with flattened columns or easily accessible structure.
    """
    # DLC CSVs usually have 3 header rows: scorer, bodyparts, coords
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    return df

def get_dlc_coord(df, frame_idx, bodypart):
    """
    Retrieves (x, y, likelihood) for a specific bodypart at a frame index.
    Searches for the bodypart in the DLC columns (handling scorer name).
    """
    # DLC columns are (scorer, bodypart, coord)
    # We just want to find the column where level 1 == bodypart
    
    # Get all columns for this bodypart
    # We assume there's only one scorer
    try:
        xs = df.xs((bodypart, 'x'), level=[1, 2], axis=1)
        ys = df.xs((bodypart, 'y'), level=[1, 2], axis=1)
        ls = df.xs((bodypart, 'likelihood'), level=[1, 2], axis=1)
        
        if xs.empty or ys.empty:
            return None, None, 0.0
            
        x = xs.iloc[frame_idx].values[0]
        y = ys.iloc[frame_idx].values[0]
        l = ls.iloc[frame_idx].values[0]
        
        return x, y, l
    except KeyError:
        return None, None, 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare DLC Inference with COCO Annotations")
    parser.add_argument("--dlc_csv", required=True, help="Path to DLC inference CSV file")
    parser.add_argument("--coco_json", required=True, help="Path to COCO annotations JSON file")
    parser.add_argument("--video_name", required=True, help="Name of the video group (e.g., 'video1_mp4') to match images")
    parser.add_argument("--output", default="comparison_results.csv", help="Output path for comparison CSV")
    
    args = parser.parse_args()

    print(f"Loading COCO annotations from {args.coco_json}...")
    coco_anns = load_coco_annotations(args.coco_json)
    
    print(f"Loading DLC predictions from {args.dlc_csv}...")
    dlc_df = load_dlc_predictions(args.dlc_csv)
    
    # -------------------------------------------------------------------------
    # Mapping Logic: match Frame Index -> Image Name
    # -------------------------------------------------------------------------
    # Logic from user's `rf_to_dlc.py`:
    #   The video was created by sorting images by frame number extracted from filename.
    #   Pattern: {video_group}-{frame_number}_jpg
    #   So we filter COCO images by `video_name` and sort them by frame number.
    
    print(f"Mapping frames for video group: {args.video_name}...")
    
    # keys are filenames
    relevant_images = []
    pattern = re.compile(r"(?P<group>.+?_mp4)-(?P<frame>\d+)_jpg")

    for img_name in coco_anns.keys():
        # Match pattern against the image filename
        # Ensure it matches the requested video_name
        # Note: image names in COCO might require checking if they contain the video name
        
        # We need to match exactly as `rf_to_dlc.py` does.
        # It expects filenames like "video1_mp4-0005_jpg.jpg" (or similar)
        # The regex provided was: r"(?P<group>.+?_mp4)-(?P<frame>\d+)_jpg"
        
        # Check if the filename itself matches the pattern
        # The regex seems to match against the *stem* or name. 
        # Let's try matching the full name.
        match = pattern.search(img_name)
        if match:
            group = match.group("group")
            if group == args.video_name:
                frame = int(match.group("frame"))
                relevant_images.append((frame, img_name))
    
    # Sort by frame number
    relevant_images.sort(key=lambda x: x[0])
    
    if not relevant_images:
        print(f"❌ No images found matching video group '{args.video_name}' in COCO annotations.")
        return

    print(f"Found {len(relevant_images)} matching images.")
    
    # -------------------------------------------------------------------------
    # Comparison Loop
    # -------------------------------------------------------------------------
    results = []
    
    # We iterate through the sorted images. 
    # The i-th image corresponds to the i-th frame in the DLC output (assuming 1-to-1).
    
    for frame_idx, (original_frame_num, img_name) in enumerate(relevant_images):
        if frame_idx >= len(dlc_df):
            print(f"⚠ Warning: More images ({len(relevant_images)}) than DLC frames ({len(dlc_df)}). Stopping.")
            break
            
        coco_kps = coco_anns[img_name]
        
        for dlc_bp, coco_bp in SUPERANIMAL_TO_ROBOFLOW.items():
            # Get Ground Truth
            # COCO keypoints might not exist for this image
            if coco_bp not in coco_kps:
                continue
                
            gt_x, gt_y, gt_v = coco_kps[coco_bp]
            
            # Skip if not visible/labeled (v=0 is usually not labeled, v=1 labeled but occluded, v=2 visible)
            # Adjust based on your specific needs. Usually v>0 means labeled.
            if gt_v == 0:
                continue
            
            # Get Prediction
            pred_x, pred_y, pred_conf = get_dlc_coord(dlc_df, frame_idx, dlc_bp)
            
            if pred_x is None:
                continue
                
            # Calc Error
            dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            
            results.append({
                "frame_idx": frame_idx,
                "image_name": img_name,
                "superanimal_part": dlc_bp,
                "coco_part": coco_bp,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "likelihood": pred_conf,
                "error": dist
            })

    # -------------------------------------------------------------------------
    # Save & Summarize
    # -------------------------------------------------------------------------
    if not results:
        print("No valid comparisons made (check bodypart names or frame mapping).")
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output, index=False)
    print(f"✓ Saved comparison results to {args.output}")
    
    print("\n=== Summary Stats ===")
    print(f"Total Keypoints Compared: {len(df_res)}")
    print(f"Mean Error (px): {df_res['error'].mean():.2f}")
    
    # Error per bodypart
    print("\nError per Bodypart:")
    summary = df_res.groupby("coco_part")["error"].describe()[['count', 'mean', 'std']]
    print(summary)


if __name__ == "__main__":
    main()
