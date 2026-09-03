from datetime import datetime
from pathlib import Path

import pytest

from app import config
from app.pipeline.video import fallback
from app.pipeline.video.base import VideoBackend
from app.queue.models import Job, Shot

from tests.conftest import make_brief


class FakeBackend(VideoBackend):
    def __init__(self, name: str, should_fail: bool = False) -> None:
        self.name = name
        self.should_fail = should_fail
        self.calls: list[tuple[Shot, Path, Path | None]] = []

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        self.calls.append((shot, output_path, reference_image))
        if self.should_fail:
            raise RuntimeError(f"{self.name} failed")
        output_path.write_bytes(b"fake video bytes")


def _job(num_shots: int = 1) -> Job:
    return Job(job_id="19082026_000000_000001", created_at=datetime.now(), brief=make_brief(num_shots))


def test_select_chain_below_threshold_uses_quality_first():
    shots = [Shot(description="a")]
    assert fallback._select_chain(shots) is fallback.CHAIN


def test_select_chain_at_threshold_uses_speed_first():
    shots = [Shot(description="a")] * config.MULTI_SHOT_THRESHOLD
    assert fallback._select_chain(shots) is fallback.MULTI_SHOT_CHAIN


def test_estimate_seconds_uses_first_model_in_selected_chain():
    shots = [Shot(description="a")] * 3
    expected = 3 * config.VIDEO_MODEL_ETA_SECONDS[fallback.CHAIN[0].name]
    assert fallback.estimate_seconds(shots) == expected


def test_estimate_seconds_multi_shot_uses_speed_first_chain():
    shots = [Shot(description="a")] * config.MULTI_SHOT_THRESHOLD
    expected = config.MULTI_SHOT_THRESHOLD * config.VIDEO_MODEL_ETA_SECONDS[fallback.MULTI_SHOT_CHAIN[0].name]
    assert fallback.estimate_seconds(shots) == expected


def test_generate_falls_back_to_next_backend_on_failure(monkeypatch, tmp_path):
    failing = FakeBackend("failing_model", should_fail=True)
    working = FakeBackend("working_model", should_fail=False)
    monkeypatch.setattr(fallback, "_select_chain", lambda shots: [failing, working])

    model_used = fallback.generate(_job(), tmp_path)

    assert model_used == "working_model"
    assert failing.calls
    assert working.calls


def test_generate_raises_when_all_backends_fail(monkeypatch, tmp_path):
    only = FakeBackend("only_model", should_fail=True)
    monkeypatch.setattr(fallback, "_select_chain", lambda shots: [only])

    with pytest.raises(RuntimeError):
        fallback.generate(_job(), tmp_path)


def test_generate_chains_reference_image_across_shots(monkeypatch, tmp_path):
    backend = FakeBackend("model")
    monkeypatch.setattr(fallback, "_select_chain", lambda shots: [backend])
    monkeypatch.setattr(fallback, "_extract_last_frame", lambda video_path, output_path: output_path)
    monkeypatch.setattr(fallback, "_concatenate", lambda clip_paths, output_path: None)

    fallback.generate(_job(num_shots=3), tmp_path)

    assert len(backend.calls) == 3
    first_ref = backend.calls[0][2]
    second_ref = backend.calls[1][2]
    third_ref = backend.calls[2][2]
    assert first_ref is None
    assert second_ref is not None
    assert third_ref is not None
