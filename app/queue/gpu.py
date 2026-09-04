import json
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum

from app import config


class GpuOverride(str, Enum):
    AUTO = "auto"
    FORCE_BLOCK = "force_block"
    FORCE_ALLOW = "force_allow"


@dataclass
class GpuReading:
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float


_override_lock = threading.Lock()


def _load_override() -> GpuOverride:
    try:
        return GpuOverride(json.loads(config.GPU_OVERRIDE_FILE.read_text())["override"])
    except (OSError, ValueError, KeyError):
        return GpuOverride.AUTO


# Persisted so "about to game, don't start anything" survives a service restart.
_override = _load_override()


def set_override(value: GpuOverride) -> None:
    global _override
    with _override_lock:
        _override = value
        try:
            config.GPU_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.GPU_OVERRIDE_FILE.write_text(json.dumps({"override": value.value}))
        except OSError:
            pass


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

    return reading.memory_used_mb <= config.GPU_IDLE_VRAM_MB
