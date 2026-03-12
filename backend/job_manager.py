"""
In-memory job manager — tracks inference jobs, their status, and log messages.

Each job has:
  - id          : unique UUID
  - status      : queued → running → completed | failed
  - progress    : float 0.0–1.0 for real-time progress tracking
  - logs        : list of timestamped log strings (pollable)
  - result      : dict returned by run_inference (only when completed)
  - error       : error message (only when failed)

Inference runs in a separate *process* (multiprocessing) so it never blocks
the main FastAPI process.  A lightweight daemon thread monitors a Queue for
progress / log / result messages from the child and updates the Job object.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Use 'spawn' so CUDA is never double-initialised via fork.
_mp_ctx = mp.get_context("spawn")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
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
            "progress": round(self.progress, 4),
            "video_name": self.video_name,
            "model_name": self.model_name,
            "superanimal_name": self.superanimal_name,
            "elapsed_seconds": elapsed,
            "logs": self.logs,
            "result_files": (self.result or {}).get("result_files", []),
            "error": self.error,
        }


# ── Picklable progress reporter (passed to the child process) ────────────────


class ProgressReporter:
    """Sends progress / log messages back to the parent via a Queue."""

    def __init__(self, queue: mp.Queue):
        self.queue = queue

    def __call__(self, progress: float, message: str = "") -> None:
        self.queue.put(("progress", progress, message))

    def log(self, message: str, level: str = "INFO") -> None:
        self.queue.put(("log", level, message))


# ── Stdout / stderr capture (child process) ─────────────────────────────────


class _StreamToQueue:
    """Redirect stdout / stderr writes into the Queue as log messages."""

    def __init__(self, queue: mp.Queue, level: str = "INFO"):
        self.queue = queue
        self.level = level
        self._buf = ""

    def write(self, text: str) -> None:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self.queue.put(("log", self.level, line))

    def flush(self) -> None:
        if self._buf.strip():
            self.queue.put(("log", self.level, self._buf.strip()))
            self._buf = ""


# ── Child-process worker (must be a top-level function for pickling) ─────────


class _QueueLogHandler(logging.Handler):
    """Send Python logging records from the child process to the parent via Queue."""

    def __init__(self, queue: mp.Queue):
        super().__init__()
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname
        message = self.format(record)
        self.queue.put(("log", level, message))


def _subprocess_worker(queue: mp.Queue, run_fn, args: tuple, kwargs: dict):
    """
    Runs inside a spawned child process.

    - Redirects stdout / stderr so all DLC / library output is captured.
    - Routes Python logging to the parent via Queue.
    - Injects a ``progress_callback`` into *kwargs* so the work function
      can report progress.
    - Sends result or error back through the queue.
    """
    # Redirect stdout / stderr
    sys.stdout = _StreamToQueue(queue, "INFO")
    sys.stderr = _StreamToQueue(queue, "WARNING")

    # Route all Python logging in the child process to the Queue
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    queue_handler = _QueueLogHandler(queue)
    queue_handler.setFormatter(logging.Formatter("%(name)s │ %(message)s"))
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    reporter = ProgressReporter(queue)
    kwargs["progress_callback"] = reporter

    try:
        result = run_fn(*args, **kwargs)
        queue.put(("result", result))
    except Exception as exc:
        queue.put(("error", str(exc)))
    finally:
        # Flush captured streams
        sys.stdout.flush()
        sys.stderr.flush()


# ── Queue monitor (runs in a daemon thread in the *main* process) ────────────


_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _monitor_queue(job: Job, process: mp.Process, queue: mp.Queue) -> None:
    """Read messages from the child until it exits, updating the Job."""
    while process.is_alive() or not queue.empty():
        try:
            msg = queue.get(timeout=0.5)
        except Exception:
            continue

        kind = msg[0]
        if kind == "progress":
            _, progress_val, message = msg
            job.progress = float(progress_val)
            if message:
                job.add_log(message)
                logger.info("[job %s] %s", job.id, message)
        elif kind == "log":
            _, level, message = msg
            job.add_log(message, level=level)
            logger.log(
                _LOG_LEVEL_MAP.get(level, logging.INFO),
                "[job %s] %s", job.id, message,
            )
        elif kind == "result":
            _, result = msg
            job.result = result
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.add_log(
                f"Inference completed — {len(result.get('result_files', []))} file(s)"
            )
            logger.info("[job %s] Completed — %d file(s)", job.id, len(result.get('result_files', [])))
        elif kind == "error":
            _, error_msg = msg
            job.status = JobStatus.FAILED
            logger.error("[job %s] FAILED: %s", job.id, error_msg)
            job.error = error_msg
            job.add_log(f"Inference FAILED: {error_msg}", level="ERROR")

    # If the process ended without sending result / error, mark as failed
    if job.status == JobStatus.RUNNING:
        exit_code = process.exitcode
        job.status = JobStatus.FAILED
        job.error = f"Worker process exited unexpectedly (code {exit_code})"
        job.add_log(job.error, level="ERROR")

    job.completed_at = time.time()


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
    Execute ``run_fn(*args, **kwargs)`` in a **child process**, updating the
    job's status and progress via a multiprocessing Queue.

    A ``progress_callback`` (a :class:`ProgressReporter`) is injected into
    *kwargs* automatically — the work function can call it with
    ``progress_callback(0.5, "Half done")`` to report progress.
    """
    queue = _mp_ctx.Queue()

    job.status = JobStatus.RUNNING
    job.started_at = time.time()
    job.add_log("Inference started (subprocess)")

    process = _mp_ctx.Process(
        target=_subprocess_worker,
        args=(queue, run_fn, args, kwargs),
        daemon=True,
    )
    process.start()

    # Lightweight daemon thread to read the queue and update the Job
    monitor = threading.Thread(
        target=_monitor_queue,
        args=(job, process, queue),
        daemon=True,
    )
    monitor.start()
