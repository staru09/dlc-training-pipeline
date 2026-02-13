# DLC Training Pipeline

Clean, modular scripts for DeepLabCut training workflow.

## Scripts

| Script            | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| `convert_data.py` | Convert Roboflow keypoint dataset to DLC format |
| `init_project.py` | Initialize a new DLC project with custom config |
| `train.py`        | Train model with SuperAnimal transfer learning  |
| `inference.py`    | Run inference on videos                         |

## Quick Start

### 1. Convert Roboflow Data

```bash
python convert_data.py --input dataset --output labeled-data --scorer aru
```

### 2. Initialize Project

```bash
python init_project.py --name dog_tracking --scorer aru --labeled-data labeled-data/milo_15s
```

### 3. Train Model

```bash
# Basic training
python train.py --config dog_tracking-aru-2025-02-04/config.yaml

# With SuperAnimal transfer learning (recommended)
python train.py --config dog_tracking-aru-2025-02-04/config.yaml --superanimal

# Custom epochs
python train.py --config dog_tracking-aru-2025-02-04/config.yaml --superanimal --epochs 100
```

### 4. Run Inference

```bash
# Analyze video
python inference.py --config dog_tracking-aru-2025-02-04/config.yaml --video test.mp4

# Create labeled video
python inference.py --config dog_tracking-aru-2025-02-04/config.yaml --video test.mp4 --create-labeled

# Batch process folder
python inference.py --config dog_tracking-aru-2025-02-04/config.yaml --video videos/ --output results/
```

## DLC Evaluation Functions

DeepLabCut provides built-in evaluation tools:

```python
import deeplabcut

# Evaluate trained model (generates plots + metrics)
deeplabcut.evaluate_network(config_path, plotting=True)

# Plot keypoint trajectories
deeplabcut.plot_trajectories(config_path, [video_path])

# Filter/smooth predictions
deeplabcut.filterpredictions(config_path, [video_path])

# Create video with keypoints overlaid
deeplabcut.create_labeled_video(config_path, [video_path])
```

The `train.py` script automatically runs `evaluate_network()` after training.

## Requirements

- Python 3.9+
- DeepLabCut 3.0+
- pandas, numpy

```bash
pip install deeplabcut
```

## Project Structure

```
project-name-scorer-date/
├── config.yaml           # Project configuration
├── labeled-data/         # Training images + annotations
│   └── video_name/
│       ├── CollectedData_scorer.csv
│       ├── CollectedData_scorer.h5
│       └── *.jpg
├── training-datasets/    # Generated training data
├── dlc-models-pytorch/   # Model checkpoints
└── evaluation-results/   # Training metrics + plots
```
