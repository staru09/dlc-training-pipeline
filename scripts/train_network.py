import argparse
import os
import deeplabcut
from pathlib import Path
import shutil
import yaml

def train_setup(project_name, data_path, author="aru", model_type="superanimal_quadruped", videotype=".jpg"):
    """
    Setup and training for a single DLC project.
    
    Args:
        project_name (str): Name of the project.
        data_path (str): Path to labeled-data (containing subfolders with images/csv/h5).
        author (str): Project author.
        model_type (str): DLC SuperAnimal model type.
        videotype (str): Image extension in labeled-data.
    """
    
    base_dir = Path(os.getcwd())
    data_path = Path(data_path).resolve()
    
    print(f"--- Starting Training Setup for {project_name} ---")
    
    # 1. Project Creation / Loading
    # Check if a project configuration already exists in the current directory matching the name
    config_path = None
    
    # Simple heuristic to find existing project config
    possible_configs = list(base_dir.glob(f"{project_name}*/config.yaml"))
    
    if possible_configs:
        config_path = possible_configs[0]
        print(f"✓ Found existing project: {config_path}")
    else:
        print(f"Creating new project: {project_name}")
        # We need at least one video to create a project, but we are using labeled-data directly.
        # DLC requires passing video paths to create_new_project appropriately or we can use a dummy.
        # However, since we have labeled-data, we can point to the images there.
        
        # Collect video folders from data_path
        video_folders = [f for f in data_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
        if not video_folders:
            print("❌ Error: No folders found in data_path. Expected labeled-data structure.")
            return

        # Use the images in the first folder as "videos" for project creation triggers
        # or better, just create the project and then overwrite the labeled-data
        
        # dlc.create_new_project requires video paths. 
        # Let's use the folder paths, DLC supports directory input often
        video_list = [str(f) for f in video_folders]
        
        try:
            config_path = deeplabcut.create_new_project(
                project_name, 
                author, 
                video_list, 
                working_directory=str(base_dir), 
                copy_videos=False, 
                multianimal=False,
                superanimal_transfer_learning=True # Enable SuperAnimal mode
            )
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            return
            
    # 2. Configure Project
    if config_path:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        # Ensure model type is set for SuperAnimal
        # The key might vary based on DLC version, but usually it involves 'identity' logic or specific weights
        # For SuperAnimal transfer learning, we often run a specific function or set init_weights.
        # But 'create_new_project(..., superanimal_transfer_learning=True)' handles most.
        
        print("✓ Project configured")
        
        # 3. Create Training Dataset
        print("Creating training dataset...")
        try:
            deeplabcut.create_training_dataset(
                config_path, 
                num_shuffles=1, 
                net_type=model_type, 
                augmenter_type='imgaug'
            )
        except Exception as e:
             # Sometimes it fails if dataset is already created
             print(f"Warning during dataset creation (might already exist): {e}")

        # 4. Train Network
        print("Starting training...")
        try:
            deeplabcut.train_network(
                config_path, 
                shuffle=1, 
                displayiters=100, 
                saveiters=1000, 
                maxiters=50000,
                allow_growth=True
            )
        except KeyboardInterrupt:
            print("Training stopped by user.")
        except Exception as e:
            print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DLC SuperAnimal model")
    parser.add_argument("--project_name", required=True, help="Name of the project")
    parser.add_argument("--data_path", required=True, help="Path to labeled-data folder")
    parser.add_argument("--model_type", default="superanimal_quadruped", help="Model type")
    
    args = parser.parse_args()
    
    train_setup(
        project_name=args.project_name,
        data_path=args.data_path,
        model_type=args.model_type
    )
