# DLC SuperAnimal Inference API

GPU-accelerated animal pose estimation API powered by DeepLabCut SuperAnimal, deployed on Google Cloud Run.

## Service URL

```
https://dlc-api-service-405737646974.europe-west4.run.app
```

---

## Available Endpoints

| Method | Path                  | Description                                                              |
| ------ | --------------------- | ------------------------------------------------------------------------ |
| `GET`  | `/`                   | Health check                                                             |
| `GET`  | `/models`             | List available models, detectors, datasets                               |
| `POST` | `/infer`              | Upload video file + run inference (multipart/form-data)                  |
| `POST` | `/infer/gcs`          | GCS-to-GCS inference (application/json, recommended for production)      |
| `GET`  | `/jobs/{job_id}`      | Poll job status and progress                                             |
| `GET`  | `/results/{filename}` | Download result files                                                    |

---

## Endpoint Details & Examples

### 1. **GET /** — Health Check

Quick liveness probe.

```bash
curl -X GET "https://dlc-api-service-405737646974.europe-west4.run.app/"
```

**Response:**

```json
{
  "status": "ok",
  "service": "dlc-inference-api"
}
```

---

### 2. **GET /models** — List Available Models

Return every model, detector, and dataset the user can choose from.

```bash
curl -X GET "https://dlc-api-service-405737646974.europe-west4.run.app/models"
```

**Response:**

```json
{
  "pose_models": [
    "resnet_50",
    "resnet_101",
    "hrnet_w18",
    "hrnet_w32",
    "hrnet_w48"
  ],
  "detectors": ["fasterrcnn_resnet50_fpn_v2"],
  "superanimal_datasets": ["superanimal_quadruped", "superanimal_topviewmouse"],
  "defaults": {
    "model_name": "hrnet_w32",
    "detector_name": "fasterrcnn_resnet50_fpn_v2",
    "superanimal_name": "superanimal_quadruped"
  }
}
```

---

### 3. **POST /infer** — Video File Upload

Upload a video file directly via `multipart/form-data`. Returns a `job_id` immediately; processing runs in the background.

**Parameters:**

| Parameter             | Required | Default                      | Description                     |
| --------------------- | -------- | ---------------------------- | ------------------------------- |
| `video`               | Yes      | —                            | Video file (MP4, MOV, AVI, MKV) |
| `superanimal_name`    | No       | `superanimal_quadruped`      | SuperAnimal dataset             |
| `model_name`          | No       | `hrnet_w32`                  | Pose estimation model           |
| `detector_name`       | No       | `fasterrcnn_resnet50_fpn_v2` | Object detector                 |
| `max_individuals`     | No       | `1`                          | Max animals in frame            |
| `pcutoff`             | No       | `0.1`                        | Confidence threshold            |
| `batch_size`          | No       | `1`                          | Pose model batch size           |
| `detector_batch_size` | No       | `1`                          | Detector batch size             |
| `device`              | No       | `auto`                       | `auto`, `cuda`, `cuda:0`, `cpu` |

**cURL Example:**

```bash
curl -X POST "https://dlc-api-service-405737646974.europe-west4.run.app/infer" \
  -F "video=@input_video.mp4" \
  -F "model_name=hrnet_w32" \
  -F "superanimal_name=superanimal_quadruped" \
  -F "max_individuals=1"
```

**Response:**

```json
{
  "message": "Inference job started — poll /jobs/{job_id} for status",
  "job_id": "a1b2c3d4e5f6",
  "video_name": "input_video.mp4",
  "model_used": "hrnet_w32",
  "detector_used": "fasterrcnn_resnet50_fpn_v2",
  "superanimal": "superanimal_quadruped",
  "result_files": []
}
```

Poll `GET /jobs/{job_id}` for progress. Download results from `GET /results/{filename}` when complete.

---

### 4. **POST /infer/gcs** — GCS-to-GCS Inference (Recommended)

Process a video stored in Google Cloud Storage. Accepts `application/json`. Downloads the input from GCS, runs inference, uploads results back to GCS, and cleans up all local files. Returns a `task_id` immediately.

**Request Body (JSON):**

| Field                 | Required | Default                      | Description                                 |
| --------------------- | -------- | ---------------------------- | ------------------------------------------- |
| `gcs_input_path`      | Yes      | —                            | GCS input path: `bucket/folder/video.mp4`   |
| `gcs_output_path`     | Yes      | —                            | GCS output path: `bucket/folder`            |
| `superanimal_name`    | No       | `superanimal_quadruped`      | SuperAnimal dataset                         |
| `model_name`          | No       | `hrnet_w32`                  | Pose estimation model                       |
| `detector_name`       | No       | `fasterrcnn_resnet50_fpn_v2` | Object detector                             |
| `max_individuals`     | No       | `1`                          | Max animals in frame                        |
| `pcutoff`             | No       | `0.1`                        | Confidence threshold                        |
| `batch_size`          | No       | `1`                          | Pose model batch size                       |
| `detector_batch_size` | No       | `1`                          | Detector batch size                         |
| `device`              | No       | `auto`                       | `auto`, `cuda`, `cuda:0`, `cpu`             |

**cURL Example — Minimal (defaults for model params):**

```bash
curl -X POST "https://dlc-api-service-405737646974.europe-west4.run.app/infer/gcs" \
  -H "Content-Type: application/json" \
  -d '{
    "gcs_input_path": "datacam_videos/processed_videos/4edec7a8-651c-4c10-a653-6b8f9535caf4.mp4",
    "gcs_output_path": "datacam_videos/test_dlc"
  }'
```

**cURL Example — Custom model:**

```bash
curl -X POST "https://dlc-api-service-405737646974.europe-west4.run.app/infer/gcs" \
  -H "Content-Type: application/json" \
  -d '{
    "gcs_input_path": "datacam_videos/processed_videos/4edec7a8-651c-4c10-a653-6b8f9535caf4.mp4",
    "gcs_output_path": "datacam_videos/test_dlc",
    "model_name": "resnet_50",
    "superanimal_name": "superanimal_topviewmouse",
    "max_individuals": 3
  }'
```

**Response:**

```json
{
  "success": true,
  "message": "Task queued successfully",
  "task_id": "a1b2c3d4e5f6"
}
```

Use the returned `task_id` to poll for progress using `GET /jobs/{task_id}`.

---

### 5. **GET /jobs/{job_id}** — Poll Job Status

Check the progress of any inference job (from `/infer` or `/infer/gcs`).

```bash
curl -X GET "https://dlc-api-service-405737646974.europe-west4.run.app/jobs/a1b2c3d4e5f6"
```

**Response (While Processing):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "progress": 0.55,
  "video_name": "4edec7a8-651c-4c10-a653-6b8f9535caf4",
  "model_name": "hrnet_w32",
  "superanimal_name": "superanimal_quadruped",
  "elapsed_seconds": 12.5,
  "logs": [
    {
      "timestamp": 1710000000.0,
      "level": "INFO",
      "message": "Inference started (subprocess)"
    }
  ],
  "result_files": [],
  "error": null
}
```

**Response (Completed):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "completed",
  "progress": 1.0,
  "video_name": "4edec7a8-651c-4c10-a653-6b8f9535caf4",
  "model_name": "hrnet_w32",
  "superanimal_name": "superanimal_quadruped",
  "elapsed_seconds": 45.3,
  "logs": [...],
  "result_files": [
    "4edec7a8-651c-4c10-a653-6b8f9535caf4_annotated.mp4",
    "4edec7a8-651c-4c10-a653-6b8f9535caf4_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5",
    "4edec7a8-651c-4c10-a653-6b8f9535caf4_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2__before_adapt.json",
    "4edec7a8-651c-4c10-a653-6b8f9535caf4_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_labeled_before_adapt.mp4"
  ],
  "error": null
}
```

**Status Values:**

| Status      | Description                              |
| ----------- | ---------------------------------------- |
| `queued`    | Job is queued, waiting to start          |
| `running`   | Inference is in progress                 |
| `completed` | Done — check `result_files` for outputs  |
| `failed`    | Failed — check `error` field for details |

---

### 6. **GET /results/{filename}** — Download Result Files

Download a result file produced by inference. Only available for `/infer` jobs (GCS jobs upload directly to GCS).

```bash
curl -O "https://dlc-api-service-405737646974.europe-west4.run.app/results/4edec7a8-651c-4c10-a653-6b8f9535caf4_annotated.mp4"
```

---

## Python Examples

### File Upload with Polling

```python
import requests
import time

SERVICE_URL = "https://dlc-api-service-405737646974.europe-west4.run.app"

# 1. Upload video
with open("input.mp4", "rb") as f:
    resp = requests.post(
        f"{SERVICE_URL}/infer",
        files={"video": ("input.mp4", f, "video/mp4")},
        data={"model_name": "hrnet_w32", "superanimal_name": "superanimal_quadruped"},
        timeout=60,
    )

job_id = resp.json()["job_id"]
print(f"Job submitted: {job_id}")

# 2. Poll for completion
while True:
    status = requests.get(f"{SERVICE_URL}/jobs/{job_id}", timeout=10).json()
    print(f"Status: {status['status']} | Elapsed: {status['elapsed_seconds']}s")

    if status["status"] == "completed":
        print(f"Done! Files: {status['result_files']}")
        break
    elif status["status"] == "failed":
        print(f"Failed: {status['error']}")
        break

    time.sleep(5)

# 3. Download results
for fname in status["result_files"]:
    r = requests.get(f"{SERVICE_URL}/results/{fname}", timeout=120)
    with open(fname, "wb") as f:
        f.write(r.content)
    print(f"Downloaded: {fname}")
```

### GCS-to-GCS with Polling

```python
import requests
import time

SERVICE_URL = "https://dlc-api-service-405737646974.europe-west4.run.app"

# 1. Submit GCS task (application/json)
resp = requests.post(
    f"{SERVICE_URL}/infer/gcs",
    json={
        "gcs_input_path": "datacam_videos/processed_videos/4edec7a8-651c-4c10-a653-6b8f9535caf4.mp4",
        "gcs_output_path": "datacam_videos/test_dlc",
    },
    timeout=30,
)
result = resp.json()
task_id = result["task_id"]
print(f"Task queued: {task_id}")

# 2. Poll for completion
while True:
    poll = requests.get(f"{SERVICE_URL}/jobs/{task_id}", timeout=30).json()
    status = poll["status"]
    elapsed = poll.get("elapsed_seconds") or 0
    print(f"Status: {status} | Elapsed: {elapsed:.1f}s")

    if status == "completed":
        print(f"Inference complete! Results uploaded to GCS: {poll['result_files']}")
        break
    elif status == "failed":
        print(f"Failed: {poll['error']}")
        break

    time.sleep(5)
```

---

## Configuration

### GCS Paths

Both `gcs_input_path` and `gcs_output_path` are required per request.

- **Input path format:** `bucket_name/folder/video_uuid.mp4`
- **Output path format:** `bucket_name/folder`

### Available Models

| Model        | Architecture        |
| ------------ | ------------------- |
| `hrnet_w32`  | HRNet-W32 (default) |
| `hrnet_w18`  | HRNet-W18           |
| `hrnet_w48`  | HRNet-W48           |
| `resnet_50`  | ResNet-50           |
| `resnet_101` | ResNet-101          |

**Detector:** `fasterrcnn_resnet50_fpn_v2`

**Datasets:** `superanimal_quadruped`, `superanimal_topviewmouse`

### Output Files

Each inference run produces:

- `*_annotated.mp4` — Annotated video with keypoints and bounding boxes (H.264 encoded)
- `*.h5` — Pose predictions (HDF5 format)
- `*_before_adapt.json` — Per-frame predictions (JSON)
- `*_labeled_before_adapt.mp4` — Raw labeled video from DeepLabCut (H.264 re-encoded)
- `*.pickle` — Full predictions (if applicable)

---

## CLI Client

The `run_inference.py` script provides a CLI for both endpoints.

### File Upload Mode

```bash
python run_inference.py --video input.mp4 --output results/
python run_inference.py --video input.mp4 --output results/ --model resnet_50
```

### GCS Mode

```bash
python run_inference.py --gcs \
  --gcs-input-path datacam_videos/processed_videos/4edec7a8-651c-4c10-a653-6b8f9535caf4.mp4 \
  --gcs-output-path datacam_videos/test_dlc \
  --api-url https://dlc-api-service-405737646974.europe-west4.run.app
```
