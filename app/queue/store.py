import json
import threading
from datetime import datetime
from pathlib import Path

from app import config
from app.queue.models import Job, format_job_id

_counter_lock = threading.Lock()


def next_job_id() -> str:
    with _counter_lock:
        counter = _read_counter() + 1
        _write_counter(counter)
        return format_job_id(datetime.now(), counter)


def _read_counter() -> int:
    if not config.COUNTER_FILE.exists():
        return 0
    return json.loads(config.COUNTER_FILE.read_text())["counter"]


def _write_counter(counter: int) -> None:
    config.COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.COUNTER_FILE.write_text(json.dumps({"counter": counter}))


def job_dir(job_id: str) -> Path:
    return config.DATA_DIR / job_id


def metadata_path(job_id: str) -> Path:
    return job_dir(job_id) / "metadata.json"


def save_job(job: Job) -> None:
    path = metadata_path(job.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job.model_dump_json(indent=2))


def load_job(job_id: str) -> Job:
    return Job.model_validate_json(metadata_path(job_id).read_text())


def list_jobs() -> list[Job]:
    if not config.DATA_DIR.exists():
        return []
    jobs = [load_job(p.parent.name) for p in config.DATA_DIR.glob("*/metadata.json")]
    return sorted(jobs, key=lambda job: int(job.job_id.rsplit("_", 1)[-1]))
