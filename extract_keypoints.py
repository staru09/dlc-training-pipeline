import json
import sys
from pathlib import Path

def extract_keypoints(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'categories' not in data:
            print("No 'categories' found in JSON.")
            return
        
        output_file = Path("dataset_exploration.txt")
        # Ensure we append to the file instead of overwriting
        with open(output_file, 'a') as out:
            out.write(f"\n{'='*40}\n")
            out.write(f"Dataset: {json_path}\n")
            
            # Count images in JSON
            image_count = len(data.get('images', []))
            out.write(f"Images in JSON: {image_count}\n")
            
            # Count files in directory
            try:
                dir_path = Path(json_path).parent
                file_count = len(list(dir_path.glob('*')))
                out.write(f"Files in directory: {file_count}\n")
            except Exception as e:
                out.write(f"Files in directory: Error counting ({str(e)})\n")
                
            out.write(f"{'='*40}\n\n")
            
            for category in data['categories']:
                cat_name = category.get('name', 'unknown')
                keypoints = category.get('keypoints', [])
                
                out.write(f"Category: {cat_name}\n")
                if keypoints:
                    out.write("Keypoints:\n")
                    for i, kp in enumerate(keypoints):
                        line = f"  {i:2d}. {kp}\n"
                        out.write(line)
                        print(line.strip())
                else:
                    out.write("  No keypoints found for this category.\n")
                out.write("\n")
        
        print(f"\nSuccessfully appended body parts to {output_file.absolute()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    json_path = r"D:\dlc\dlc-training-pipeline\main\new dataset\train\_annotations.coco.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    extract_keypoints(json_path)
