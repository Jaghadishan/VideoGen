from datetime import datetime

import pytest

from app.pipeline.video.base import build_shot_prompt
from app.pipeline.video.ltx import LTX23
from app.queue.models import Job, Shot

from tests.conftest import make_brief


def _job() -> Job:
    return Job(job_id="19082026_000000_000001", created_at=datetime.now(), brief=make_brief())


class _FakeResult:
    def __init__(self):
        self.frames = [["frame"]]


class _FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult()


def test_text_to_video_uses_t2v_pipe(monkeypatch, tmp_path):
    t2v, i2v = _FakePipe(), _FakePipe()
    monkeypatch.setattr(LTX23, "_load", lambda self: (t2v, i2v))
    monkeypatch.setattr("diffusers.utils.export_to_video", lambda frames, path, fps: __import__("pathlib").Path(path).write_bytes(b"v"))

    out = tmp_path / "shot.mp4"
    LTX23().generate(_job(), Shot(description="a wide desert vista"), out)

    assert len(t2v.calls) == 1 and len(i2v.calls) == 0
    assert "a wide desert vista" in t2v.calls[0]["prompt"]
    assert "image" not in t2v.calls[0]
    assert out.exists()


def test_image_to_video_uses_i2v_pipe_with_reference(monkeypatch, tmp_path):
    t2v, i2v = _FakePipe(), _FakePipe()
    monkeypatch.setattr(LTX23, "_load", lambda self: (t2v, i2v))
    monkeypatch.setattr("diffusers.utils.export_to_video", lambda frames, path, fps: __import__("pathlib").Path(path).write_bytes(b"v"))
    monkeypatch.setattr("diffusers.utils.load_image", lambda p: f"IMG({p})")

    ref = tmp_path / "last.png"
    ref.write_bytes(b"PNG")
    LTX23().generate(_job(), Shot(description="continues the pan"), tmp_path / "shot.mp4", reference_image=ref)

    assert len(i2v.calls) == 1 and len(t2v.calls) == 0
    assert i2v.calls[0]["image"] == f"IMG({ref})"


def test_release_vram_noop_when_not_loaded():
    assert LTX23._pipe is None
    LTX23().release_vram()  # must not raise
