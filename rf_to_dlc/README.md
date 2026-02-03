# rf_to_dlc 🐕 → 🤖

Convert **Roboflow** annotated pose estimation datasets to **DeepLabCut** compatible format.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [CLI Reference](#cli-reference)
- [Programmatic Usage](#programmatic-usage)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- 📥 Download and extract Roboflow datasets
- 🎬 Create videos from image frame sequences
- 🔄 Convert COCO keypoint annotations to DLC format
- 📁 Organize images into DLC project structure
- 👁️ Visualize and verify annotations
- 🐾 Support for both **single-animal** and **multi-animal** projects
- 📊 Convert CSV to HDF5 format

---

## 🛠️ Installation

### 1. Clone or download the package

```bash
cd z:\Current Working\Nannie_ai\dlc-work
```

### 2. Install dependencies

```bash
pip install -r rf_to_dlc/requirements.txt
```

### 3. Verify installation

```bash
python -m rf_to_dlc --help
```

---

## 🚀 Quick Start

Run the complete pipeline with a single command:

```bash
python -m rf_to_dlc full-pipeline \
    --url "https://app.roboflow.com/ds/YOUR_DATASET_KEY?key=YOUR_API_KEY" \
    --output-dir ./dataset \
    --project-dir ./DLC_PROJECT \
    --video-name "milo_15s" \
    --scorer "j"
```

---

## 📖 Step-by-Step Guide

### Step 1: Export from Roboflow

1. Go to your Roboflow project
2. Navigate to **Generate** → **Versions**
3. Select **COCO Keypoints** format
4. Click **Export Dataset** → **Get Link**
5. Copy the download URL (looks like `https://app.roboflow.com/ds/...?key=...`)

### Step 2: Download Dataset

```bash
python -m rf_to_dlc download \
    --url "https://app.roboflow.com/ds/YOUR_URL?key=YOUR_KEY" \
    --output-dir ./dataset
```

**What this does:**

- Downloads the zip file from Roboflow
- Extracts to `./dataset/` with subfolders: `train/`, `valid/`, `test/`
- Each subfolder contains images and `_annotations.coco.json`

### Step 3: (Optional) Create Videos

If you want to reconstruct videos from the extracted frames:

```bash
python -m rf_to_dlc create-videos \
    --input-dir ./dataset \
    --output-dir ./dataset/videos
```

**What this does:**

- Groups images by their source video name
- Stitches frames back into `.mp4` videos at 24 fps

### Step 4: Convert Annotations

**For single-animal project:**

```bash
python -m rf_to_dlc convert \
    --input-dir ./dataset \
    --output ./CollectedData_j.csv \
    --scorer "j" \
    --image-prefix "labeled-data/milo_15s" \
    --report-missing
```

**For multi-animal project:**

```bash
python -m rf_to_dlc convert \
    --input-dir ./dataset \
    --output ./CollectedData_j.csv \
    --scorer "j" \
    --multi-animal \
    --individuals "dog1,dog2" \
    --image-prefix "labeled-data/milo_15s"
```

**What this does:**

- Parses `_annotations.coco.json` from each subfolder
- Extracts keypoint coordinates (x, y)
- Creates DLC-compatible CSV with proper header structure
- Removes duplicate images
- Optionally reports images with missing keypoints

### Step 5: Visualize and Verify

Before proceeding, verify your annotations are mapped correctly:

```bash
python -m rf_to_dlc visualize \
    --csv ./CollectedData_j.csv \
    --image-dir ./dataset \
    --samples 5
```

**What this does:**

- Samples random images
- Plots keypoints overlaid on images
- Allows you to visually verify annotation accuracy

### Step 6: Create DLC Project

Create a new DeepLabCut project (if you haven't already):

```python
import deeplabcut

config_path = deeplabcut.create_new_project(
    project='MyPoseProject',
    experimenter='j',
    videos=[],  # We'll add frames directly
    working_directory='./dlc_projects',
    copy_videos=False,
    multianimal=False  # Set True for multi-animal
)

print(f"Project created at: {config_path}")
```

### Step 7: Copy Files to DLC Project

```bash
# Copy the CollectedData CSV
python -m rf_to_dlc copy-images \
    --csv ./CollectedData_j.csv \
    --source-dir ./dataset \
    --dest-dir "./dlc_projects/MyPoseProject-j-2025-02-03/labeled-data/milo_15s"
```

Then manually copy the CSV:

```bash
copy CollectedData_j.csv "dlc_projects\MyPoseProject-j-2025-02-03\labeled-data\milo_15s\CollectedData_j.csv"
```

### Step 8: Convert CSV to HDF5

```bash
python -m rf_to_dlc to-hdf5 \
    --csv "./dlc_projects/MyPoseProject-j-2025-02-03/labeled-data/milo_15s/CollectedData_j.csv"
```

**For multi-animal:**

```bash
python -m rf_to_dlc to-hdf5 \
    --csv ./CollectedData_j.csv \
    --multi-animal
```

### Step 9: Update DLC Config

Edit your `config.yaml` to include the bodyparts from your dataset:

```yaml
bodyparts:
  - nose
  - left_eye
  - right_eye
  - left_ear
  - right_ear
  # ... add all your keypoints

skeleton:
  - - nose
    - left_eye
  - - nose
    - right_eye
  # ... define connections
```

### Step 10: Create Training Dataset

```python
import deeplabcut

config_path = "path/to/config.yaml"

# Create training dataset
deeplabcut.create_training_dataset(config_path)

# Train the model
deeplabcut.train_network(config_path)
```

---

## 📚 CLI Reference

| Command         | Description                            |
| --------------- | -------------------------------------- |
| `download`      | Download and extract Roboflow dataset  |
| `create-videos` | Create videos from image sequences     |
| `convert`       | Convert COCO annotations to DLC format |
| `copy-images`   | Copy images to DLC project folder      |
| `visualize`     | Visualize annotations on sample images |
| `frame-order`   | Create frame order metadata CSV        |
| `to-hdf5`       | Convert CollectedData CSV to HDF5      |
| `full-pipeline` | Run the complete workflow              |

### Get help for any command:

```bash
python -m rf_to_dlc <command> --help
```

---

## 🐍 Programmatic Usage

```python
from rf_to_dlc import (
    download_and_extract,
    parse_all_annotations,
    create_single_animal_headers,
    format_dataframe,
    remove_duplicates,
    save_collected_data,
    copy_images_to_project,
    get_image_names_from_df,
    verify_keypoint_mapping,
)
from pathlib import Path

# 1. Download
dataset_dir = download_and_extract(
    url="https://app.roboflow.com/ds/...",
    output_dir=Path("./dataset")
)

# 2. Parse annotations
df, keypoints = parse_all_annotations(
    base_dir=dataset_dir,
    image_prefix="labeled-data/milo_15s"
)
print(f"Found {len(df)} annotations with keypoints: {keypoints}")

# 3. Format for DLC
headers = create_single_animal_headers("j", keypoints)
df_formatted = format_dataframe(df, headers)
df_unique = remove_duplicates(df_formatted, header_rows=3)

# 4. Save
output_dir = Path("./dlc_project/labeled-data/milo_15s")
csv_path = output_dir / "CollectedData_j.csv"
save_collected_data(df_unique, csv_path)

# 5. Copy images
image_names = get_image_names_from_df(df_unique, header_rows=3)
copy_images_to_project(image_names, dataset_dir, output_dir)

# 6. Verify
verify_keypoint_mapping(csv_path, dataset_dir, num_samples=5)
```

---

## 🔧 Troubleshooting

### "Scorer name mismatch" error

Ensure the `scorer` in your `config.yaml` matches the scorer name used in `CollectedData_j.csv`:

```yaml
# config.yaml
scorer: j # Must match the CSV
```

### "DataFrame structure mismatch" error

This usually means the header format is wrong. Check:

- Single-animal: 3 header rows (scorer, bodyparts, coords)
- Multi-animal: 4 header rows (scorer, individuals, bodyparts, coords)

### Keypoints not showing up

Check the visibility mode. By default, only fully visible keypoints (`v=2`) are included. To include occluded keypoints:

```python
from rf_to_dlc import Config

config = Config(visibility_mode=1)  # Include labeled but occluded
```

### Missing keypoints report

```bash
python -m rf_to_dlc convert --input-dir ./dataset --output ./data.csv --report-missing
```

This creates `data_missing_keypoints.csv` listing problematic annotations.

---

## 📁 Project Structure

```
rf_to_dlc/
├── __init__.py           # Package exports
├── __main__.py           # Module entry point
├── config.py             # Configuration dataclass
├── utils.py              # Shared utilities
├── downloader.py         # Dataset download/extraction
├── video_creator.py      # Frame → video conversion
├── annotation_parser.py  # COCO → DLC parsing
├── data_formatter.py     # DLC header formatting
├── file_manager.py       # File copying/organization
├── visualizer.py         # Annotation visualization
├── cli.py                # Command-line interface
├── requirements.txt      # Dependencies
└── README.md             # This file
```

---

## 📝 License

MIT License - Feel free to use and modify!
