#!/usr/bin/env python3
"""
Client script to run inference via the FastAPI backend.

Usage (file upload):
    python run_inference.py --video /path/to/input.mp4 --output /path/to/output_dir
    python run_inference.py --video input.mp4 --output results/ --model resnet_50

Usage (GCS-to-GCS):
    python run_inference.py --gcs --video-name my_video.mp4
    python run_inference.py --gcs --video-name my_video.mp4 --gcs-input-path other_bucket/folder
    python run_inference.py --gcs --video-name my_video.mp4 --gcs-output-path dlc_bucket/output
"""

import argparse
import sys
import time
from pathlib import Path

import requests

DEFAULT_API_URL = "http://localhost:8000"
POLL_INTERVAL = 1  # seconds


def run(
    video_path: str,
    output_dir: str,
    api_url: str = DEFAULT_API_URL,
    model_name: str = "hrnet_w32",
    detector_name: str = "fasterrcnn_resnet50_fpn_v2",
    superanimal_name: str = "superanimal_quadruped",
    max_individuals: int = 1,
    pcutoff: float = 0.1,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    device: str = "auto",
):
    video = Path(video_path).resolve()
    out = Path(output_dir).resolve()

    if not video.is_file():
        print(f"ERROR: Video not found → {video}")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Health check ──────────────────────────────────────────────────
    try:
        r = requests.get(f"{api_url}/", timeout=5)
        r.raise_for_status()
        print(f"✓ API is live at {api_url}")
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach API at {api_url}. Is the server running?")
        sys.exit(1)

    # ── 2. Send video for inference ──────────────────────────────────────
    print(f"⬆ Uploading {video.name} ({video.stat().st_size / 1e6:.1f} MB) ...")
    print(f"  model={model_name}  detector={detector_name}  dataset={superanimal_name}")

    with open(video, "rb") as f:
        resp = requests.post(
            f"{api_url}/infer",
            files={"video": (video.name, f, "video/mp4")},
            data={
                "superanimal_name": superanimal_name,
                "model_name": model_name,
                "detector_name": detector_name,
                "max_individuals": max_individuals,
                "pcutoff": pcutoff,
                "batch_size": batch_size,
                "detector_batch_size": detector_batch_size,
                "device": device,
            },
            timeout=60,
        )

    if resp.status_code != 200:
        print(f"ERROR ({resp.status_code}): {resp.text}")
        sys.exit(1)

    result = resp.json()
    job_id = result["job_id"]
    print(f"✓ Job submitted — job_id: {job_id}")

    # ── 3. Poll for status ───────────────────────────────────────────────
    _poll_job(api_url, job_id)

    # ── 4. Download results ──────────────────────────────────────────────
    status_resp = requests.get(f"{api_url}/jobs/{job_id}", timeout=10)
    job_data = status_resp.json()
    result_files = job_data.get("result_files", [])
    print(f"\n📥 Downloading {len(result_files)} result file(s) ...")

    for fname in result_files:
        print(f"  ⬇ {fname} ...", end=" ")
        dl = requests.get(f"{api_url}/results/{fname}", timeout=120)
        if dl.status_code == 200:
            dest = out / fname
            dest.write_bytes(dl.content)
            print(f"→ {dest}")
        else:
            print(f"FAILED ({dl.status_code})")

    print(f"\n✅ Done! Results saved to {out}")


def run_gcs(
    video_name: str,
    api_url: str = DEFAULT_API_URL,
    gcs_input_path: str | None = None,
    gcs_output_path: str | None = None,
    model_name: str = "hrnet_w32",
    detector_name: str = "fasterrcnn_resnet50_fpn_v2",
    superanimal_name: str = "superanimal_quadruped",
    max_individuals: int = 1,
    pcutoff: float = 0.1,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    device: str = "auto",
):
    """Submit a GCS-to-GCS inference task and poll for completion."""

    # ── 1. Health check ──────────────────────────────────────────────────
    try:
        r = requests.get(f"{api_url}/", timeout=5)
        r.raise_for_status()
        print(f"✓ API is live at {api_url}")
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach API at {api_url}. Is the server running?")
        sys.exit(1)

    # ── 2. Submit GCS task ───────────────────────────────────────────────
    data = {
        "video_name": video_name,
        "superanimal_name": superanimal_name,
        "model_name": model_name,
        "detector_name": detector_name,
        "max_individuals": max_individuals,
        "pcutoff": pcutoff,
        "batch_size": batch_size,
        "detector_batch_size": detector_batch_size,
        "device": device,
    }
    if gcs_input_path:
        data["gcs_input_path"] = gcs_input_path
    if gcs_output_path:
        data["gcs_output_path"] = gcs_output_path

    print(f"🚀 Submitting GCS task for {video_name} ...")
    if gcs_input_path:
        print(f"  Input: {gcs_input_path}/{video_name}")
    if gcs_output_path:
        print(f"  Output: {gcs_output_path}/")

    try:
        resp = requests.post(f"{api_url}/infer/gcs", data=data, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Failed to submit task: {e}")
        sys.exit(1)

    result = resp.json()
    task_id = result["task_id"]
    print(f"✓ Task queued — task_id: {task_id}")

    # ── 3. Poll for completion ───────────────────────────────────────────
    _poll_job(api_url, task_id)

    print(f"\n✅ Done! Results uploaded to GCS.")


def _poll_job(api_url: str, job_id: str):
    """Poll a job until completion or failure."""
    print(f"\n🔄 Polling job status (every {POLL_INTERVAL}s) ...\n")
    seen_logs = 0

    while True:
        try:
            status_resp = requests.get(f"{api_url}/jobs/{job_id}", timeout=10)
            status_resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ⚠ Poll error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        job_data = status_resp.json()
        status = job_data["status"]
        logs = job_data.get("logs", [])
        elapsed = job_data.get("elapsed_seconds")

        # Print any new log lines
        for log_entry in logs[seen_logs:]:
            level = log_entry.get("level", "INFO")
            msg = log_entry.get("message", "")
            prefix = "❌" if level == "ERROR" else "  📋"
            print(f"{prefix} [{level}] {msg}")
        seen_logs = len(logs)

        if status == "completed":
            elapsed_str = f" in {elapsed}s" if elapsed else ""
            print(f"\n✅ Inference completed{elapsed_str}")
            return
        elif status == "failed":
            error = job_data.get("error", "Unknown error")
            print(f"\n❌ Inference FAILED: {error}")
            sys.exit(1)

        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(
        description="Run SuperAnimal inference via the FastAPI backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples (file upload):
  python run_inference.py --video video.mp4 --output results/
  python run_inference.py --video video.mp4 --output results/ --model resnet_50

Examples (GCS-to-GCS):
  python run_inference.py --gcs --video-name my_video.mp4
  python run_inference.py --gcs --video-name my_video.mp4 --gcs-input-path other_bucket/folder
        """,
    )
    # Mode selection
    parser.add_argument("--gcs", action="store_true",
                        help="Use GCS-to-GCS mode (uses /infer/gcs endpoint)")

    # File upload mode args
    parser.add_argument("--video", "-v", help="Path to input video (file upload mode)")
    parser.add_argument("--output", "-o", help="Directory to save results (file upload mode)")

    # GCS mode args
    parser.add_argument("--video-name", help="Video filename in GCS bucket (GCS mode)")
    parser.add_argument("--gcs-input-path", default=None,
                        help="GCS input path as bucket/folder (default: server config)")
    parser.add_argument("--gcs-output-path", default=None,
                        help="GCS output path as bucket/folder (default: server config)")

    # Common args
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--model", default="hrnet_w32", choices=[
        "resnet_50", "resnet_101", "hrnet_w18", "hrnet_w32", "hrnet_w48",
    ], help="Pose model (default: hrnet_w32)")
    parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2", help="Detector name")
    parser.add_argument("--superanimal", default="superanimal_quadruped",
                        choices=["superanimal_quadruped", "superanimal_topviewmouse"],
                        help="SuperAnimal dataset (default: superanimal_quadruped)")
    parser.add_argument("--max-individuals", type=int, default=1, help="Max animals in frame")
    parser.add_argument("--pcutoff", type=float, default=0.1, help="Confidence cutoff")
    parser.add_argument("--batch-size", type=int, default=1, help="Pose model batch size")
    parser.add_argument("--detector-batch-size", type=int, default=1, help="Detector batch size")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")

    args = parser.parse_args()

    if args.gcs:
        if not args.video_name:
            parser.error("--video-name is required in GCS mode")
        run_gcs(
            video_name=args.video_name,
            api_url=args.api_url,
            gcs_input_path=args.gcs_input_path,
            gcs_output_path=args.gcs_output_path,
            model_name=args.model,
            detector_name=args.detector,
            superanimal_name=args.superanimal,
            max_individuals=args.max_individuals,
            pcutoff=args.pcutoff,
            batch_size=args.batch_size,
            detector_batch_size=args.detector_batch_size,
            device=args.device,
        )
    else:
        if not args.video or not args.output:
            parser.error("--video and --output are required in file upload mode")
        run(
            video_path=args.video,
            output_dir=args.output,
            api_url=args.api_url,
            model_name=args.model,
            detector_name=args.detector,
            superanimal_name=args.superanimal,
            max_individuals=args.max_individuals,
            pcutoff=args.pcutoff,
            batch_size=args.batch_size,
            detector_batch_size=args.detector_batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
