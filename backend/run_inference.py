#!/usr/bin/env python3
"""
Client script to run inference via the FastAPI backend.

Usage:
    python run_inference.py --video /path/to/input.mp4 --output /path/to/output_dir
    python run_inference.py --video input.mp4 --output results/ --model resnet_50
    python run_inference.py --video input.mp4 --output results/ --superanimal superanimal_topviewmouse
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
            ts = log_entry.get("timestamp", "")
            level = log_entry.get("level", "INFO")
            msg = log_entry.get("message", "")
            prefix = "❌" if level == "ERROR" else "  📋"
            print(f"{prefix} [{level}] {msg}")
        seen_logs = len(logs)

        if status == "completed":
            elapsed_str = f" in {elapsed}s" if elapsed else ""
            print(f"\n✅ Inference completed{elapsed_str}")
            break
        elif status == "failed":
            error = job_data.get("error", "Unknown error")
            print(f"\n❌ Inference FAILED: {error}")
            sys.exit(1)

        time.sleep(POLL_INTERVAL)

    # ── 4. Download results ──────────────────────────────────────────────
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


def main():
    parser = argparse.ArgumentParser(
        description="Run SuperAnimal inference via the FastAPI backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_inference.py --video video.mp4 --output results/
  python run_inference.py --video video.mp4 --output results/ --model resnet_50
  python run_inference.py --video video.mp4 --output results/ --superanimal superanimal_topviewmouse --max-individuals 3
        """,
    )
    parser.add_argument("--video", "-v", required=True, help="Path to input video")
    parser.add_argument("--output", "-o", required=True, help="Directory to save results")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--model", default="hrnet_w32", choices=[
        "resnet_50", "resnet_101", "hrnet_w18", "hrnet_w32", "hrnet_w48",
    ], help="Pose model (default: hrnet_w32)")
    parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2", help="Detector name")
    parser.add_argument("--superanimal", default="superanimal_quadruped",
                        choices=["superanimal_quadruped", "superanimal_topviewmouse"],
                        help="SuperAnimal dataset (default: superanimal_quadruped)")
    parser.add_argument("--max-individuals", type=int, default=1, help="Max animals in frame (default: 1)")
    parser.add_argument("--pcutoff", type=float, default=0.1, help="Confidence cutoff (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=1, help="Pose model batch size")
    parser.add_argument("--detector-batch-size", type=int, default=1, help="Detector batch size")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")

    args = parser.parse_args()

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
