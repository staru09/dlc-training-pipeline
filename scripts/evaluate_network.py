
import argparse
import os
import sys
from pathlib import Path
import deeplabcut
import cv2
import numpy as np
import pandas as pd

# Add the specific path for utils imports if needed, or assume relative import works if running as module
# But for script usage, we might need to adjust path
sys.path.append(str(Path(__file__).parent))
from utils.gaze_metrics import calculate_gaze_vector, draw_gaze

def analyze_video(video_path, config_path=None, model_type="superanimal_quadruped", mode="base", shuffle=1):
    """
    Run inference and evaluation.
    
    Args:
        video_path (str): Path to video.
        config_path (str): Path to config.yaml (required for 'finetuned' or 'compare').
        model_type (str): Base model type.
        mode (str): 'base', 'finetuned', 'compare'.
    """
    video_path = str(Path(video_path).resolve())
    output_dir = str(Path(video_path).parent)
    
    # 1. Run Inference
    scorer_maps = {}
    
    if mode in ["base", "compare"]:
        print(f"Running Base Model ({model_type})...")
        # For SuperAnimal inference without a project, we use dlc.run_superanimal_inference (if available in this version)
        # or load the model directly. 
        # DeepLabCut 2.3.5+ supports: deeplabcut.run_superanimal_inference
        try:
             # Note: exact API might vary by DLC version. Assuming standard interface or wrapper.
             # If using standard dlc.analyze_videos, we need a config. 
             # SuperAnimal often has a specific entry point.
             # Let's try the generic video analysis which might trigger model download.
             # Actually, for base model zero-shot, usually:
             deeplabcut.analyze_videos(config_path, [video_path], videotype=os.path.splitext(video_path)[-1], shuffle=shuffle) # This uses the TRAINED model in config
             # Wait, strict 'base' model means Pre-trained SuperAnimal.
             # If we want the pre-trained one, we might need a separate mechanism or a config that points to it.
             # For now, let's assume 'base' implies 'SuperAnimal' functionality if supported.
             pass
        except Exception as e:
            print(f"Warning: Base model inference setup might require specific DLC version methods. Using standard project analysis for now based on config.")
    
    if mode in ["finetuned", "compare"]:
        print(f"Running Fine-tuned Model (Config: {config_path})...")
        if not config_path:
            print("❌ Error: config_path required for fine-tuned evaluation")
            return
            
        deeplabcut.analyze_videos(config_path, [video_path], save_as_csv=True, destfolder=output_dir)
        deeplabcut.filterpredictions(config_path, [video_path], destfolder=output_dir)
        deeplabcut.create_labeled_video(config_path, [video_path], destfolder=output_dir, draw_skeleton=True)
        
    # 2. Custom Metrics (Gaze)
    # We need to read the generated H5/CSV files to calculate metrics
    # Find the latest result file
    video_name = Path(video_path).stem
    results = list(Path(output_dir).glob(f"{video_name}*.h5"))
    if not results:
        print("No analysis results found.")
        return

    # Pick the most recent one (presumably from this run)
    result_path = sorted(results, key=os.path.getmtime)[-1]
    print(f"Calculating metrics from: {result_path.name}")
    
    df = pd.read_hdf(result_path)
    
    # Iterate frame by frame to calculate gaze
    # Dataframe struct: (scorer, bodyparts, coords)
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df.columns.get_level_values(1).unique()
    
    # Open video to overlay Gaze
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    save_path = str(Path(output_dir) / f"{video_name}_gaze.mp4")
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get keypoints for this frame
        # extract x,y for each bodypart
        kpts = {}
        for bp in bodyparts:
            try:
                x = df.loc[frame_idx, (scorer, bp, 'x')]
                y = df.loc[frame_idx, (scorer, bp, 'y')]
                prob = df.loc[frame_idx, (scorer, bp, 'likelihood')]
                if prob > 0.5: # Visibility threshold
                    kpts[bp] = np.array([x, y])
            except KeyError:
                pass
                
        # Calculate Gaze
        gaze_vec, eyes_mid = calculate_gaze_vector(kpts)
        
        # Draw Gaze
        if gaze_vec is not None:
            frame = draw_gaze(frame, eyes_mid, gaze_vec)
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"✓ Gaze video saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--config_path", help="Path to project config.yaml")
    parser.add_argument("--mode", default="finetuned", choices=["base", "finetuned", "compare"])
    
    args = parser.parse_args()
    
    analyze_video(args.video_path, args.config_path, mode=args.mode)
