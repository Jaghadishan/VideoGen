import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.queue import gpu, heartbeat

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_override():
    yield
    gpu.set_override(gpu.GpuOverride.AUTO)


def test_status_reports_worker_and_idle_gpu(monkeypatch):
    monkeypatch.setattr(heartbeat, "is_alive", lambda: True)
    monkeypatch.setattr(gpu, "read_gpu", lambda: gpu.GpuReading(600.0, 12000.0, 3.0))

    body = client.get("/status").json()

    assert body["worker_alive"] is True
    assert body["gpu"]["reading_ok"] is True
    assert body["gpu"]["available"] is True
    assert body["gpu"]["memory_used_mb"] == 600.0
    assert body["gpu"]["override"] == "auto"


def test_status_gpu_in_use_when_busy(monkeypatch):
    monkeypatch.setattr(heartbeat, "is_alive", lambda: True)
    monkeypatch.setattr(gpu, "read_gpu", lambda: gpu.GpuReading(8000.0, 12000.0, 90.0))

    assert client.get("/status").json()["gpu"]["available"] is False


def test_status_handles_missing_worker_and_no_gpu(monkeypatch):
    monkeypatch.setattr(heartbeat, "is_alive", lambda: False)
    monkeypatch.setattr(gpu, "read_gpu", lambda: None)

    body = client.get("/status").json()

    assert body["worker_alive"] is False
    assert body["gpu"]["reading_ok"] is False
    assert body["gpu"]["available"] is False
    assert body["gpu"]["memory_used_mb"] is None


def test_force_block_override_wins_over_idle_gpu(monkeypatch):
    monkeypatch.setattr(heartbeat, "is_alive", lambda: True)
    monkeypatch.setattr(gpu, "read_gpu", lambda: gpu.GpuReading(100.0, 12000.0, 0.0))

    body = client.post("/gpu/override", json={"override": "force_block"}).json()

    assert body["gpu"]["override"] == "force_block"
    assert body["gpu"]["available"] is False
    assert gpu.get_override() is gpu.GpuOverride.FORCE_BLOCK


def test_force_allow_override_wins_over_busy_gpu(monkeypatch):
    monkeypatch.setattr(heartbeat, "is_alive", lambda: True)
    monkeypatch.setattr(gpu, "read_gpu", lambda: gpu.GpuReading(9000.0, 12000.0, 99.0))

    body = client.post("/gpu/override", json={"override": "force_allow"}).json()

    assert body["gpu"]["available"] is True


def test_override_is_persisted(monkeypatch):
    from app import config

    client.post("/gpu/override", json={"override": "force_block"})
    assert config.GPU_OVERRIDE_FILE.exists()
    assert gpu._load_override() is gpu.GpuOverride.FORCE_BLOCK


def test_override_rejects_unknown_value():
    assert client.post("/gpu/override", json={"override": "nonsense"}).status_code == 422
