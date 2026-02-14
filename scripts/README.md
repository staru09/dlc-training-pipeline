# DeepLabCut Automation Scripts

This folder contains consolidated scripts to automate the DeepLabCut (DLC) workflow for training and evaluation.

## Scripts Overview

| Script                | Purpose                                                                  |
| :-------------------- | :----------------------------------------------------------------------- |
| `convert_data.py`     | Converts Roboflow/COCO annotations to DLC format.                        |
| `train_network.py`    | Creates a DLC project and runs the training loop.                        |
| `evaluate_network.py` | Runs inference and calculates custom metrics (Gaze, Euclidean distance). |

## 1. Data Conversion (`convert_data.py`)

If you have data exported from Roboflow (COCO Keypoint format), run this first.

```bash
python scripts/convert_data.py --input path/to/dataset --output path/to/labeled-data
```

**Key Arguments:**

- `--input`: Folder containing `train/`, `valid/`, `test/` subfolders from Roboflow.
- `--output`: Destination for the DLC-ready `labeled-data`.
- `--scorer`: Name of the scorer (default: `aru`).

## 2. Training (`train_network.py`)

Trains a SuperAnimal model on your dataset.

```bash
python scripts/train_network.py --project_name "MyProject" --data_path path/to/labeled-data
```

**Key Arguments:**

- `--project_name`: Name for your DLC project.
- `--data_path`: Path to the `labeled-data` folder (output of step 1).
- `--model_type`: SuperAnimal model to use (default: `superanimal_quadruped`).
- `--videotype`: Image extension (default: `.jpg`).

**Configuration:**

- The script automatically updates `config.yaml` to include your videos.
- It sets `max_iters` to 50,000 by default (change in script if needed).

## 3. Evaluation (`evaluate_network.py`)

Runs inference on a video and calculates Gaze.

```bash
python scripts/evaluate_network.py --video_path path/to/video.mp4 --mode compare --config_path path/to/project/config.yaml
```

**Key Arguments:**

- `--video_path`: Path to the video or directory of videos to analyze.
- `--mode`:
  - `base`: Use the pre-trained SuperAnimal model (Zero-shot).
  - `finetuned`: Use your trained model (requires `--config_path`).
  - `compare`: Run both and generate a comparison video.
- `--config_path`: Path to the `config.yaml` of your trained project.

---

## Customization

### Modify Gaze Logic

Edit `scripts/utils/gaze_metrics.py` to change how "Gaze" is calculated. Currently, it is defined as the vector from the **midpoint of the eyes** to the **nose**.

### Change Training Parameters

Open `scripts/train_network.py` and modify the `train_network` function call:

- `maxiters`: Number of training iterations.
- `saveiters`: How often to save snapshots.
