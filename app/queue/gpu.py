import subprocess
import threading
from dataclasses import dataclass
from enum import Enum

IDLE_VRAM_MB = 500.0


class GpuOverride(str, Enum):
    AUTO = "auto"
    FORCE_BLOCK = "force_block"
    FORCE_ALLOW = "force_allow"


@dataclass
class GpuReading:
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float


_override = GpuOverride.AUTO
_override_lock = threading.Lock()


def set_override(value: GpuOverride) -> None:
    global _override
    with _override_lock:
        _override = value


def get_override() -> GpuOverride:
    with _override_lock:
        return _override


def read_gpu() -> GpuReading | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    used, total, util = output.strip().split(", ")
    return GpuReading(memory_used_mb=float(used), memory_total_mb=float(total), utilization_pct=float(util))


def gpu_available() -> bool:
    override = get_override()
    if override == GpuOverride.FORCE_BLOCK:
        return False
    if override == GpuOverride.FORCE_ALLOW:
        return True

    reading = read_gpu()
    if reading is None:
        return False

    return reading.memory_used_mb <= IDLE_VRAM_MB
