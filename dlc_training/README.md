# dlc_training 🐕 → 🤖

A modular package for DeepLabCut model training workflows.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [CLI Reference](#cli-reference)
- [Programmatic Usage](#programmatic-usage)

---

## 🛠️ Installation

The package uses the existing DeepLabCut installation:

```bash
pip install --pre deeplabcut
```

---

## 🚀 Quick Start

### Complete Training Workflow

```python
from dlc_training import (
    create_project,
    check_labels,
    convert_csv_to_h5,
    create_training_dataset,
    train_network,
    evaluate_network,
    analyze_videos,
    create_labeled_video,
)

# 1. Create project
config = create_project(
    project_name="DogPose",
    experimenter="john",
    videos=["./videos/dog1.mp4"]
)

# 2. After labeling frames, convert to HDF5
convert_csv_to_h5(config, scorer="john")

# 3. Create training dataset
create_training_dataset(
    config,
    net_type="resnet_50",
    augmenter_type="albumentations"
)

# 4. Train
train_network(config, epochs=200, batch_size=8)

# 5. Evaluate
evaluate_network(config, shuffles=[1])

# 6. Analyze new videos
analyze_videos(config, videos=["./test_video.mp4"])

# 7. Create labeled video
create_labeled_video(config, videos=["./test_video.mp4"])
```

---

## 📦 Modules

| Module           | Description                           |
| ---------------- | ------------------------------------- |
| `config.py`      | `TrainingConfig` dataclass            |
| `project.py`     | Project creation and label management |
| `dataset.py`     | Training dataset creation             |
| `trainer.py`     | Network training and fine-tuning      |
| `evaluator.py`   | Model evaluation                      |
| `analyzer.py`    | Video analysis and outlier detection  |
| `visualizer.py`  | Labeled videos and trajectory plots   |
| `superanimal.py` | SuperAnimal transfer learning         |
| `utils.py`       | Debug utilities                       |

---

## 💻 CLI Reference

```bash
# Create project
python -m dlc_training create-project --name MyProject --experimenter john

# Create dataset
python -m dlc_training create-dataset --config ./config.yaml --net-type resnet_50

# Train
python -m dlc_training train --config ./config.yaml --epochs 200

# Evaluate
python -m dlc_training evaluate --config ./config.yaml --shuffles 1

# Analyze videos
python -m dlc_training analyze --config ./config.yaml --videos ./video.mp4

# Create labeled video
python -m dlc_training create-video --config ./config.yaml --videos ./video.mp4 --skeleton
```

---

## 🐾 SuperAnimal Transfer Learning

```python
from dlc_training import train_with_transfer_learning, train_with_memory_replay

# Basic transfer learning
train_with_transfer_learning(
    config_path="./config.yaml",
    from_shuffle=1,
    target_shuffle=2,
    superanimal_name="superanimal_quadruped",
    model_name="hrnet_w32",
    epochs=50
)

# Memory replay (prevents forgetting)
train_with_memory_replay(
    config_path="./config.yaml",
    from_shuffle=1,
    target_shuffle=3,
    superanimal_name="superanimal_quadruped",
    epochs=50
)
```

---

## 🔧 Debugging Utilities

```python
from dlc_training import check_image_sizes, check_video_sizes, debug_annotation_data

# Check image dimensions
check_image_sizes("./labeled-data/")

# Check video dimensions
check_video_sizes(["./video1.mp4", "./video2.mp4"])

# Debug annotation merging issues
df = debug_annotation_data("./config.yaml")
```

---

## 📁 Project Structure

```
dlc_training/
├── __init__.py        # Package exports
├── __main__.py        # CLI entry point
├── config.py          # Configuration dataclass
├── project.py         # Project management
├── dataset.py         # Dataset creation
├── trainer.py         # Network training
├── evaluator.py       # Model evaluation
├── analyzer.py        # Video analysis
├── visualizer.py      # Visualization
├── superanimal.py     # Transfer learning
├── utils.py           # Utilities
├── cli.py             # Command-line interface
└── README.md          # This file
```
