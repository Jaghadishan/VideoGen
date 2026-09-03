import subprocess

import pytest

from app.queue import gpu


@pytest.fixture(autouse=True)
def reset_override():
    yield
    gpu.set_override(gpu.GpuOverride.AUTO)


def _reading(memory_used_mb: float) -> gpu.GpuReading:
    return gpu.GpuReading(memory_used_mb=memory_used_mb, memory_total_mb=12000, utilization_pct=0)


def test_force_block_overrides_reading(monkeypatch):
    monkeypatch.setattr(gpu, "read_gpu", lambda: _reading(0))
    gpu.set_override(gpu.GpuOverride.FORCE_BLOCK)
    assert gpu.gpu_available() is False


def test_force_allow_overrides_reading(monkeypatch):
    monkeypatch.setattr(gpu, "read_gpu", lambda: _reading(9000))
    gpu.set_override(gpu.GpuOverride.FORCE_ALLOW)
    assert gpu.gpu_available() is True


def test_auto_mode_available_when_idle(monkeypatch):
    monkeypatch.setattr(gpu, "read_gpu", lambda: _reading(100))
    gpu.set_override(gpu.GpuOverride.AUTO)
    assert gpu.gpu_available() is True


def test_auto_mode_unavailable_when_busy(monkeypatch):
    monkeypatch.setattr(gpu, "read_gpu", lambda: _reading(5000))
    gpu.set_override(gpu.GpuOverride.AUTO)
    assert gpu.gpu_available() is False


def test_auto_mode_fails_safe_when_reading_unavailable(monkeypatch):
    monkeypatch.setattr(gpu, "read_gpu", lambda: None)
    gpu.set_override(gpu.GpuOverride.AUTO)
    assert gpu.gpu_available() is False


def test_read_gpu_returns_none_on_subprocess_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise subprocess.SubprocessError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "check_output", raise_error)
    assert gpu.read_gpu() is None
