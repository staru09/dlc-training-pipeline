# DLC SuperAnimal Inference API

FastAPI backend for running DeepLabCut SuperAnimal model inference on videos.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** DeepLabCut and PyTorch must already be installed in your environment.

## Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs at: `http://localhost:8000/docs`

## Run Inference

### Using the client script

```bash
# Default model (hrnet_w32)
python run_inference.py --video /path/to/video.mp4 --output /path/to/results/

# Try a different model
python run_inference.py --video video.mp4 --output results/ --model resnet_50

# TopViewMouse dataset, 3 animals
python run_inference.py --video video.mp4 --output results/ \
    --superanimal superanimal_topviewmouse --max-individuals 3
```

### Using curl

```bash
curl -X POST http://localhost:8000/infer \
  -F "video=@video.mp4" \
  -F "model_name=hrnet_w32" \
  -F "superanimal_name=superanimal_quadruped" \
  -F "max_individuals=1"
```

### Download results

```bash
curl -O http://localhost:8000/results/<filename>
```

## API Endpoints

| Method | Path                  | Description                                |
| ------ | --------------------- | ------------------------------------------ |
| `GET`  | `/`                   | Health check                               |
| `GET`  | `/models`             | List available models, detectors, datasets |
| `POST` | `/infer`              | Upload video + run inference               |
| `GET`  | `/results/{filename}` | Download result files                      |

## Available Models

| Model        | Architecture        |
| ------------ | ------------------- |
| `hrnet_w32`  | HRNet-W32 (default) |
| `hrnet_w18`  | HRNet-W18           |
| `hrnet_w48`  | HRNet-W48           |
| `resnet_50`  | ResNet-50           |
| `resnet_101` | ResNet-101          |

**Detector:** `fasterrcnn_resnet50_fpn_v2`

**Datasets:** `superanimal_quadruped`, `superanimal_topviewmouse`

## Output Files

Each inference run produces:

- `*_annotated.mp4` — Video with keypoints and bounding boxes overlaid
- `*.h5` — Pose predictions (HDF5)
- `*.json` — Per-frame predictions (JSON)
- `*.pickle` — Full predictions

## Project Structure

```
backend/
├── main.py           # FastAPI app & endpoints
├── inference.py      # Core inference service
├── annotator.py      # Custom OpenCV video annotator
├── config.py         # Settings & model registry
├── schemas.py        # Pydantic request/response models
├── run_inference.py  # Client script
├── requirements.txt  # Python dependencies
├── uploads/          # Uploaded videos (auto-created)
└── outputs/          # Inference results (auto-created)
```

## Parameters

| Parameter             | Default                      | Description                     |
| --------------------- | ---------------------------- | ------------------------------- |
| `superanimal_name`    | `superanimal_quadruped`      | Dataset                         |
| `model_name`          | `hrnet_w32`                  | Pose model                      |
| `detector_name`       | `fasterrcnn_resnet50_fpn_v2` | Object detector                 |
| `max_individuals`     | `1`                          | Max animals in frame            |
| `pcutoff`             | `0.1`                        | Confidence threshold            |
| `batch_size`          | `1`                          | Pose model batch size           |
| `detector_batch_size` | `1`                          | Detector batch size             |
| `device`              | `auto`                       | `auto`, `cuda`, `cuda:0`, `cpu` |

## Google Cloud Storage (GCS)

If configured via `DLC_GCS_OUTPUT_BUCKET`, inference result files will be automatically uploaded to GCS.

```bash
# Example setting locally
export DLC_GCS_OUTPUT_BUCKET="gs://dlc_bucket/dlc_output_main"
uvicorn main:app
```

The `cloudbuild.yaml` handles this environment injection for Cloud Run.
