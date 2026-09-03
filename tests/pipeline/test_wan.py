import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from app import config
from app.pipeline.video.wan import Wan22TI2V5B
from app.queue.models import Job, Shot

from tests.conftest import make_brief


def _job() -> Job:
    return Job(job_id="19082026_000000_000001", created_at=datetime.now(), brief=make_brief())


def _setup(monkeypatch, tmp_path):
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    wan_python = tmp_path / "python.exe"
    wan_python.write_text("")
    script = tmp_path / "wan_infer.py"
    script.write_text("")
    monkeypatch.setattr(config, "WAN_TI2V_5B_PATH", model_dir)
    monkeypatch.setattr(config, "WAN_PYTHON", wan_python)
    monkeypatch.setattr(config, "WAN_INFER_SCRIPT", script)
    return model_dir


def test_generate_errors_when_not_set_up(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WAN_TI2V_5B_PATH", tmp_path / "missing")
    monkeypatch.setattr(config, "WAN_PYTHON", tmp_path / "missing/python.exe")
    with pytest.raises(FileNotFoundError, match="Wan 2.2 TI2V-5B not set up"):
        Wan22TI2V5B().generate(_job(), Shot(description="a shot"), tmp_path / "out.mp4")


def test_text_to_video_command_has_no_image_flag(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"vid")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out = tmp_path / "out.mp4"
    Wan22TI2V5B().generate(_job(), Shot(description="a calm shot of the sea"), out)

    cmd = captured["cmd"]
    assert "--image" not in cmd
    assert cmd[cmd.index("--prompt") + 1].endswith("calm")  # mood_and_style tail from make_brief
    assert "a calm shot of the sea" in cmd[cmd.index("--prompt") + 1]
    assert cmd[cmd.index("--frames") + 1] == str(config.WAN_NUM_FRAMES)


def test_image_to_video_passes_reference_frame(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    ref = tmp_path / "last_frame.png"
    ref.write_bytes(b"PNG")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"vid")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    Wan22TI2V5B().generate(_job(), Shot(description="continues"), tmp_path / "out.mp4", reference_image=ref)

    cmd = captured["cmd"]
    assert "--image" in cmd
    assert cmd[cmd.index("--image") + 1] == str(ref.resolve())


def test_nonzero_exit_raises_with_stderr_tail(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "CUDA out of memory\nboom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        Wan22TI2V5B().generate(_job(), Shot(description="x"), tmp_path / "out.mp4")
