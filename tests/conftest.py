import pytest

from app import config
from app.queue.models import AudioType, Brief, ContentPolicy, Shot


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "COUNTER_FILE", tmp_path / "counter.json")
    monkeypatch.setattr(config, "GPU_OVERRIDE_FILE", tmp_path / "gpu_override.json")


def make_brief(num_shots: int = 1) -> Brief:
    return Brief(
        title="Test video",
        visual_description="A test scene.",
        shots=[Shot(description=f"shot {i}") for i in range(num_shots)],
        audio_type=AudioType.NONE,
        script_or_lyrics="",
        script_was_provided=True,
        mood_and_style="calm",
        target_length="short",
        content_policy=ContentPolicy.STANDARD,
    )
