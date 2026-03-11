# DLC SuperAnimal Inference API

GPU-accelerated animal pose estimation API powered by DeepLabCut SuperAnimal, deployed on Google Cloud Run.

## 🌐 Service URL

```
https://<your-cloud-run-service-url>
```

Replace with your actual deployed Cloud Run service URL.

---

## 📋 Available Endpoints

| Method | Path                  | Description                                                              |
| ------ | --------------------- | ------------------------------------------------------------------------ |
| `GET`  | `/`                   | Health check                                                             |
| `GET`  | `/models`             | List available models, detectors, datasets                               |
| `POST` | `/infer`              | Upload video file + run inference (synchronous upload, async processing) |
| `POST` | `/infer/gcs`          | GCS-to-GCS inference (async, recommended for production)                 |
| `GET`  | `/jobs/{job_id}`      | Poll job status and progress                                             |
| `GET`  | `/results/{filename}` | Download result files                                                    |

---

## 🔍 Endpoint Details & Examples

### 1. **GET /** — Health Check

Quick liveness probe.

```bash
curl -X GET "https://<service-url>/"
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
curl -X GET "https://<service-url>/models"
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

### 3. **POST /infer** — Video File Upload (Async Processing)

Upload a video file directly. Returns a `job_id` immediately; processing runs in the background.

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
curl -X POST "https://<service-url>/infer" \
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

**Note:** Poll `GET /jobs/{job_id}` for progress. Download results from `GET /results/{filename}` when complete. If `DLC_GCS_OUTPUT_BUCKET` is set, results are also uploaded to GCS automatically.

---

### 4. **POST /infer/gcs** — GCS-to-GCS Inference (Async, Recommended)

Process a video stored in Google Cloud Storage. Downloads the input from GCS, runs inference, uploads results back to GCS, and cleans up all local files. Returns a `job_id` immediately.

**Parameters:**

| Parameter             | Required | Default                                       | Description                                                     |
| --------------------- | -------- | --------------------------------------------- | --------------------------------------------------------------- |
| `video_name`          | Yes      | —                                             | Video filename in the GCS input bucket                          |
| `gcs_input_path`      | No       | `DLC_GCS_INPUT_PATH` env var (`datacam_videos/processed_videos`) | GCS input path as `bucket/folder`              |
| `gcs_output_path`     | No       | `DLC_GCS_OUTPUT_BUCKET` env var               | GCS output path as `bucket/folder`                              |
| `superanimal_name`    | No       | `superanimal_quadruped`                       | SuperAnimal dataset                                             |
| `model_name`          | No       | `hrnet_w32`                                   | Pose estimation model                                           |
| `detector_name`       | No       | `fasterrcnn_resnet50_fpn_v2`                  | Object detector                                                 |
| `max_individuals`     | No       | `1`                                           | Max animals in frame                                            |
| `pcutoff`             | No       | `0.1`                                         | Confidence threshold                                            |
| `batch_size`          | No       | `1`                                           | Pose model batch size                                           |
| `detector_batch_size` | No       | `1`                                           | Detector batch size                                             |
| `device`              | No       | `auto`                                        | `auto`, `cuda`, `cuda:0`, `cpu`                                 |

**cURL Example — Minimal (just video name, uses defaults):**

```bash
curl -X POST "https://<service-url>/infer/gcs" \
  -F "video_name=my_video.mp4"
```

→ Reads from `gs://datacam_videos/processed_videos/my_video.mp4`
→ Writes to `gs://dlc_bucket/dlc_output_main/`

**cURL Example — Custom input/output paths:**

```bash
curl -X POST "https://<service-url>/infer/gcs" \
  -F "video_name=my_video.mp4" \
  -F "gcs_input_path=other_bucket/other_folder" \
  -F "gcs_output_path=dlc_bucket/dlc_output_main"
```

**cURL Example — Custom model:**

```bash
curl -X POST "https://<service-url>/infer/gcs" \
  -F "video_name=my_video.mp4" \
  -F "model_name=resnet_50" \
  -F "superanimal_name=superanimal_topviewmouse" \
  -F "max_individuals=3"
```

**Response (Immediate):**

```json
{
  "success": true,
  "message": "Task queued successfully",
  "task_id": "a1b2c3d4e5f6"
}
```

**Note:** Use the returned `task_id` to poll for progress using `GET /jobs/{task_id}`.

---

### 5. **GET /jobs/{job_id}** — Poll Job Status

Check the progress of any inference job (from `/infer` or `/infer/gcs`).

```bash
curl -X GET "https://<service-url>/jobs/a1b2c3d4e5f6"
```

**Response (While Processing):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "video_name": "005d0bf7-446c-4a06-867d-5b41e0aa468c.mkv",
  "model_name": "hrnet_w32",
  "superanimal_name": "superanimal_quadruped",
  "elapsed_seconds": 12.5,
  "logs": [
    {
      "timestamp": 1710000000.0,
      "level": "INFO",
      "message": "Inference started"
    },
    {
      "timestamp": 1710000005.0,
      "level": "INFO",
      "message": "Starting inference | video=..."
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
  "video_name": "005d0bf7-446c-4a06-867d-5b41e0aa468c.mkv",
  "model_name": "hrnet_w32",
  "superanimal_name": "superanimal_quadruped",
  "elapsed_seconds": 45.3,
  "logs": [...],
  "result_files": [
    "005d0bf7_video_annotated.mp4",
    "005d0bf7_video_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5",
    "005d0bf7_video_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2__before_adapt.json"
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
curl -O "https://<service-url>/results/005d0bf7_video_annotated.mp4"
```

---

## 🐍 Python Examples

### Example 1: File Upload with Polling

```python
import requests
import time

SERVICE_URL = "https://<service-url>"

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
        print(f"✅ Done! Files: {status['result_files']}")
        break
    elif status["status"] == "failed":
        print(f"❌ Failed: {status['error']}")
        break

    time.sleep(5)

# 3. Download results
for fname in status["result_files"]:
    r = requests.get(f"{SERVICE_URL}/results/{fname}", timeout=120)
    with open(fname, "wb") as f:
        f.write(r.content)
    print(f"Downloaded: {fname}")
```

### Example 2: GCS-to-GCS Async Processing with Polling

```python
import requests
import time

SERVICE_URL = "https://<service-url>"

# 1. Check health
health = requests.get(f"{SERVICE_URL}/").json()
print(f"API Status: {health['status']}")

# 2. Submit GCS task (only video_name required — input/output paths use defaults)
resp = requests.post(
    f"{SERVICE_URL}/infer/gcs",
    data={
        "video_name": "my_video.mp4",
    },
    timeout=30,
)
result = resp.json()
task_id = result["task_id"]
print(f"✅ Task queued: {task_id}")

# 3. Poll for completion
for attempt in range(240):  # 20 minutes max
    poll = requests.get(f"{SERVICE_URL}/jobs/{task_id}", timeout=30).json()
    status = poll["status"]
    elapsed = poll.get("elapsed_seconds") or 0

    print(f"Status: {status} | Elapsed: {elapsed:.1f}s")

    if status == "completed":
        print(f"\n✅ Inference complete!")
        print(f"Result files uploaded to GCS: {poll['result_files']}")
        break
    elif status == "failed":
        print(f"\n❌ Failed: {poll['error']}")
        break

    time.sleep(5)
```

---

## 🎬 Common Use Cases

### Use Case 1: Process Video from GCS (Recommended)

```bash
# Step 1: Submit task (only video_name required)
RESPONSE=$(curl -s -X POST "https://<service-url>/infer/gcs" \
  -F "video_name=my_video.mp4")

JOB_ID=$(echo $RESPONSE | jq -r '.task_id')
echo "Job ID: $JOB_ID"

# Step 2: Poll for completion
while true; do
  POLL=$(curl -s "https://<service-url>/jobs/$JOB_ID")
  STATUS=$(echo $POLL | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ]; then
    echo "✅ Done!"
    echo $POLL | jq '.result_files'
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "❌ Failed"
    echo $POLL | jq -r '.error'
    exit 1
  fi

  sleep 5
done
```

### Use Case 2: Direct File Upload

```bash
curl -X POST "https://<service-url>/infer" \
  -F "video=@dog_playing.mp4" \
  -F "model_name=hrnet_w32" \
  -F "superanimal_name=superanimal_quadruped"
```

---

## ⚙️ Configuration

### Environment Variables

| Variable                | Description                          | Example                              |
| ----------------------- | ------------------------------------ | ------------------------------------ |
| `DLC_GCS_INPUT_PATH`    | Default GCS input path (bucket/folder) | `datacam_videos/processed_videos`  |
| `DLC_GCS_OUTPUT_BUCKET` | Default GCS output path for results  | `gs://dlc_bucket/dlc_output_main`    |

Both can be overridden per request via `gcs_input_path` and `gcs_output_path` parameters.

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

- `*_annotated.mp4` — Video with keypoints and bounding boxes overlaid
- `*.h5` — Pose predictions (HDF5)
- `*.json` — Per-frame predictions (JSON)
- `*.pickle` — Full predictions

---

## 📦 GCS Bucket Storage

For `/infer/gcs` jobs, results are automatically uploaded to GCS and local files are cleaned up.

**Default output:** `gs://dlc_bucket/dlc_output_main/`

**Per-request:** Use `gcs_output_path` parameter (format: `bucket/folder`)

**List outputs:**

```bash
gsutil ls gs://dlc_bucket/dlc_output_main/
```

**Download from GCS:**

```bash
gsutil cp gs://dlc_bucket/dlc_output_main/005d0bf7_video_annotated.mp4 ./
```
