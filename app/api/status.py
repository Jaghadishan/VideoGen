from fastapi import APIRouter
from pydantic import BaseModel

from app.queue import gpu, heartbeat

router = APIRouter(tags=["status"])


class GpuStatus(BaseModel):
    available: bool
    override: gpu.GpuOverride
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    utilization_pct: float | None = None
    reading_ok: bool


class Status(BaseModel):
    worker_alive: bool
    gpu: GpuStatus


class OverrideRequest(BaseModel):
    override: gpu.GpuOverride


def _status() -> Status:
    reading = gpu.read_gpu()
    return Status(
        worker_alive=heartbeat.is_alive(),
        gpu=GpuStatus(
            available=gpu.gpu_available(),
            override=gpu.get_override(),
            reading_ok=reading is not None,
            memory_used_mb=reading.memory_used_mb if reading else None,
            memory_total_mb=reading.memory_total_mb if reading else None,
            utilization_pct=reading.utilization_pct if reading else None,
        ),
    )


@router.get("/status", response_model=Status)
def get_status() -> Status:
    return _status()


@router.post("/gpu/override", response_model=Status)
def set_gpu_override(request: OverrideRequest) -> Status:
    gpu.set_override(request.override)
    return _status()
