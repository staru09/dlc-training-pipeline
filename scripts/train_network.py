import argparse
import os
import deeplabcut
from pathlib import Path
import shutil
import yaml
import pandas as pd

import sys
import subprocess
import traceback

# Import conversion function from convert_data.py
sys.path.insert(0, str(Path(__file__).parent))
from convert_data import convert_to_dlc_format

def setup_model_variant(base_project_name, data_path, model_type, author="aru", videotype=".jpg"):
    """
    Setup a specific model variant (Create Project -> Sync Config -> Create Dataset).
    Returns: config_path or None
    """
    # Unique project name for this variant
    project_name = f"{base_project_name}_{model_type}"
    base_dir = Path(os.getcwd())
    data_path = Path(data_path).resolve()
    
    print(f"\n[{model_type}] Setting up project: {project_name}")
    
    # 1. Project Creation / Loading
    config_path = None
    possible_configs = list(base_dir.glob(f"{project_name}*/config.yaml"))
    
    if possible_configs:
        config_path = possible_configs[0]
        print(f"[{model_type}] ✓ Found existing project: {config_path}")
    else:
        video_folders = [f for f in data_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
        if not video_folders:
            print(f"[{model_type}] ❌ Error: No folders found in data_path.")
            return None

        # video_list = [str(f) for f in video_folders]
        # Passing folders to create_new_project causes it to scan for images and add them individually
        # We want to treat folders as "labeled-data" sources. 
        # Strategy: Create project using ONE dummy file (to satisfy DLC requirements), 
        # then manually symlink/copy folders into labeled-data.
        
        # Find a dummy file (jpg/png) in the first folder
        dummy_video = None
        for vid_folder in video_folders:
            for ext in ['*.jpg', '*.png', '*.mp4', '*.avi']:
                try:
                    found = next(vid_folder.glob(ext))
                    dummy_video = str(found.resolve())
                    break
                except StopIteration:
                    continue
            if dummy_video:
                break
                
        if not dummy_video:
            print(f"[{model_type}] ❌ Error: No valid images/videos found in data folders to initialize project.")
            return None

        try:
            config_path = deeplabcut.create_new_project(
                project_name, 
                author, 
                [dummy_video], # Pass single dummy file
                working_directory=str(base_dir), 
                copy_videos=False, 
                multianimal=False
            )
            
            # Manual Data Setup
            project_dir = Path(config_path).parent
            labeled_data_dir = project_dir / "labeled-data"
            labeled_data_dir.mkdir(exist_ok=True)
            
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                
            cfg['video_sets'] = {}
            
            for folder in video_folders:
                dest_folder = labeled_data_dir / folder.name
                
                # Copy or Symlink
                if dest_folder.exists():
                    shutil.rmtree(dest_folder) # Clean cleanup if needed
                
                # Try symlink first, fall back to copy
                try:
                    # Windows requires admin for symlinks usually, copy is safer for general use
                    if os.name == 'nt':
                        shutil.copytree(folder, dest_folder)
                    else:
                        os.symlink(folder, dest_folder)
                except OSError:
                    shutil.copytree(folder, dest_folder)
                
                # Auto-convert raw COCO datasets if needed
                has_collected = list(dest_folder.glob("CollectedData_*.csv"))
                has_coco = (dest_folder / "_annotations.coco.json").exists()
                if not has_collected and has_coco:
                    print(f"[{model_type}] Converting COCO data in {folder.name}...")
                    convert_to_dlc_format(
                        input_dir=dest_folder,
                        output_dir=labeled_data_dir,
                        scorer=author,
                        video_name=folder.name
                    )
                elif has_collected:
                    print(f"[{model_type}] ✓ {folder.name} already has CollectedData files")
                    
                # Register in config
                cfg['video_sets'][str(dest_folder.resolve())] = {'crop': '0, 100, 0, 100'} 
                
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
                
            print(f"[{model_type}] ✓ Linked data folders: {[f.name for f in video_folders]}")

        except Exception as e:
            print(f"[{model_type}] ❌ Error creating project: {e}")
            return None
            
    # 2. Configure Project
    if config_path:
        # Sync Bodyparts
        try:
            # Look in the potentially linked/copied folders now
            project_dir = Path(config_path).parent
            csv_files = list((project_dir / "labeled-data").rglob("CollectedData*.csv"))
            
            if csv_files:
                sample_csv = csv_files[0]
                df = pd.read_csv(sample_csv, header=[0, 1, 2], index_col=0, nrows=0)
                bodyparts = list(df.columns.get_level_values(1).unique())
                
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                
                cfg['bodyparts'] = bodyparts
                with open(config_path, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                # print(f"[{model_type}] ✓ Synced bodyparts")
        except Exception as e:
            print(f"[{model_type}] ⚠ Warning: Failed to sync bodyparts: {e}")

        # 3. Create Training Dataset
        try:
            deeplabcut.create_training_dataset(
                config_path, 
                num_shuffles=1, 
                net_type=model_type, 
                augmenter_type='imgaug'
            )
        except Exception as e:
             # pass # Often fails if already exists, harmless
             print(f"[{model_type}] ⚠ create_training_dataset warning: {e}") 
             traceback.print_exc()

    return config_path

def train_variant(config_path, model_type, maxiters=50):
    """
    Run the actual training loop.
    """
    print(f"\n[{model_type}] Starting training for {maxiters} iterations...")
    try:
        deeplabcut.train_network(
            config_path, 
            shuffle=1, 
            displayiters=10, 
            saveiters=maxiters if maxiters < 500 else 500, 
            maxiters=maxiters,
            allow_growth=True
        )
    except KeyboardInterrupt:
        print(f"[{model_type}] Training stopped by user.")
    except Exception as e:
        print(f"[{model_type}] ❌ Error during training: {e}")

def main_setup(project_name, data_path, specific_model=None, parallel=False, maxiters=50):
    models_to_train = [
        "resnet_50", 
        "resnet_101", 
        "hrnet_w18", 
        "hrnet_w32", 
        "hrnet_w48"
    ]
    
    if specific_model:
        models_to_train = [specific_model]
        
    # Phase 1: Setup all projects sequentially
    # (DLC project creation might not be thread-safe for config editing etc)
    configs = {}
    print("--- Phase 1: Setup ---")
    for model in models_to_train:
        path = setup_model_variant(project_name, data_path, model)
        if path:
            configs[model] = path
            
    # Phase 2: Training
    print("\n--- Phase 2: Training ---")
    if parallel and len(configs) > 1:
        print(f"🚀 Launching {len(configs)} training jobs in PARALLEL on GPU...")
        processes = []
        for model, config_path in configs.items():
            # Launch separate python process for each training to isolate memory/context
            # We call this script again with a special flag
            cmd = [
                sys.executable, 
                __file__, 
                "--train_only_config", str(config_path),
                "--model_type", model,
                "--maxiters", str(maxiters)
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
            print(f"   [Started] {model} (PID: {p.pid})")
            
        # Wait for completion
        for p in processes:
            p.wait()
        print("✓ All parallel training jobs completed.")
    else:
        # Sequential
        for model, config_path in configs.items():
            train_variant(config_path, model, maxiters=maxiters)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DLC models")
    # Main usage args
    parser.add_argument("--project_name", help="Base name of the project")
    parser.add_argument("--data_path", help="Path to labeled-data folder")
    parser.add_argument("--model", help="Run only a specific model")
    parser.add_argument("--parallel", action="store_true", help="Run training in parallel")
    parser.add_argument("--maxiters", type=int, default=50, help="Number of training iterations (default: 50)")
    
    # Internal worker args
    parser.add_argument("--train_only_config", help="Internal: Path to config for worker process")
    parser.add_argument("--model_type", help="Internal: Model type name")
    
    args = parser.parse_args()
    
    if args.train_only_config:
        # Worker mode
        train_variant(args.train_only_config, args.model_type, maxiters=args.maxiters)
    elif args.project_name and args.data_path:
        # Main mode
        main_setup(
            project_name=args.project_name,
            data_path=args.data_path,
            specific_model=args.model,
            parallel=args.parallel,
            maxiters=args.maxiters
        )
    else:
        parser.print_help()
