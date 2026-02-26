"""
In-memory job manager — tracks inference jobs, their status, and log messages.

Each job has:
  - id          : unique UUID
  - status      : queued → running → completed | failed
  - logs        : list of timestamped log strings (pollable)
  - result      : dict returned by run_inference (only when completed)
  - error       : error message (only when failed)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    logs: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    # Parameters for display
    video_name: str = ""
    model_name: str = ""
    superanimal_name: str = ""

    def add_log(self, message: str, level: str = "INFO") -> None:
        """Append a timestamped log entry."""
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
        })

    def to_dict(self) -> dict:
        """Serialise for the API response."""
        elapsed = None
        if self.started_at:
            end = self.completed_at or time.time()
            elapsed = round(end - self.started_at, 1)

        return {
            "job_id": self.id,
            "status": self.status.value,
            "video_name": self.video_name,
            "model_name": self.model_name,
            "superanimal_name": self.superanimal_name,
            "elapsed_seconds": elapsed,
            "logs": self.logs,
            "result_files": (self.result or {}).get("result_files", []),
            "error": self.error,
        }


class _JobLogHandler(logging.Handler):
    """
    A logging handler that captures log records and appends them to a Job.
    Attach this handler while a job is running so *all* library output
    (including DLC's own prints) is captured.
    """

    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.job.add_log(msg, level=record.levelname)
        except Exception:
            pass


# ── Global job store (in-memory, single-process) ────────────────────────────

_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(video_name: str, model_name: str, superanimal_name: str) -> Job:
    """Create a new job and register it."""
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        video_name=video_name,
        model_name=model_name,
        superanimal_name=superanimal_name,
    )
    with _lock:
        _jobs[job_id] = job
    logger.info("Created job %s", job_id)
    return job


def get_job(job_id: str) -> Job | None:
    """Retrieve a job by ID."""
    with _lock:
        return _jobs.get(job_id)


def run_job_in_background(
    job: Job,
    run_fn,
    *args,
    **kwargs,
) -> None:
    """
    Execute `run_fn(*args, **kwargs)` in a background thread, updating the
    job's status and capturing logs.
    """

    def _worker():
        # Attach a log handler so all logging output is captured into the job
        root_logger = logging.getLogger()
        handler = _JobLogHandler(job)
        handler.setFormatter(logging.Formatter("%(name)s │ %(message)s"))
        root_logger.addHandler(handler)

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.add_log("Inference started")

        try:
            result = run_fn(*args, **kwargs)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.add_log(
                f"Inference completed — {len(result.get('result_files', []))} file(s)"
            )
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.add_log(f"Inference FAILED: {exc}", level="ERROR")
            logger.exception("Job %s failed", job.id)
        finally:
            job.completed_at = time.time()
            root_logger.removeHandler(handler)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
