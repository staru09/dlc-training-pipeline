# DeepLabCut Automation Scripts

This folder contains consolidated scripts to automate the DeepLabCut (DLC) workflow for training and evaluation.

## Scripts Overview

| Script                | Purpose                                                                      |
| :-------------------- | :--------------------------------------------------------------------------- |
| `convert_data.py`     | Converts Roboflow/COCO annotations to DLC format (Smart Category Selection). |
| `train_network.py`    | Automates creation and training of **5 distinct model architectures**.       |
| `evaluate_network.py` | Runs inference and calculates custom metrics (Gaze, Euclidean distance).     |

## 1. Data Conversion (`convert_data.py`)

If you have data exported from Roboflow (COCO Keypoint format), run this first.

```bash
python scripts/convert_data.py --input path/to/dataset --output path/to/labeled-data --video-name my_video_name
```

**Key Features:**

- **Smart Category Selection**: Automatically finds the "dog" category (or the one with the most keypoints) to avoid selecting "gaze" or empty categories.
- **Validation**: Checks for missing images and validates keypoint names against the expected 18-keypoint dog skeleton (`right_eye`, `nose`, etc.).

## 2. Training (`train_network.py`)

Automatically creates and trains **5 separate DLC projects** for different architectures using your dataset.

**Architectures Trained:**

1.  `resnet_50`
2.  `resnet_101`
3.  `hrnet_w18`
4.  `hrnet_w32`
5.  `hrnet_w48`

**Usage:**

Train all 5 models (Sequentially):

```bash
python scripts/train_network.py --project_name "Dataset1" --data_path labeled-data/
```

Train all 5 models **in PARALLEL** (Optimized for A100):

```bash
python scripts/train_network.py --project_name "Dataset1" --data_path labeled-data/ --parallel
```

_This launches 5 concurrent training jobs. Ensure your GPU has enough VRAM (A100 is great for this)._

Train a specific model only:

```bash
python scripts/train_network.py --project_name "Dataset1" --data_path labeled-data/ --model hrnet_w32
```

**Configuration:**

- **Auto-Sync**: The script reads your `CollectedData_*.csv` to find the exact bodyparts (e.g., `right_eye`, `nose`) and updates `config.yaml` automatically.
- **Transfer Learning**: Uses standard transfer learning (ImageNet weights) for these architectures.

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

Edit `scripts/utils/gaze_metrics.py`.

- **Logic**: Vector from **Midpoint(Left Eye, Right Eye)** to **Nose**.
- **Mapping**: Automatically handles `left_eye` -> `lefteye` mapping.

### Change Training Parameters

Open `scripts/train_network.py` and modify the `train_network` function call:

- `maxiters`: Number of training iterations (Default: 50,000).
- `saveiters`: How often to save snapshots.
