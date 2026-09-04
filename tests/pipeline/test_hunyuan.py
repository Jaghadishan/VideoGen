import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from app import config
from app.pipeline.video.hunyuan import HunyuanVideo15
from app.queue.models import Job, Shot

from tests.conftest import make_brief


def _job() -> Job:
    return Job(job_id="19082026_000000_000001", created_at=datetime.now(), brief=make_brief())


def _setup(monkeypatch, tmp_path):
    model_dir = tmp_path / "hunyuan"
    model_dir.mkdir()
    wan_python = tmp_path / "python.exe"
    wan_python.write_text("")
    script = tmp_path / "hunyuan_infer.py"
    script.write_text("")
    monkeypatch.setattr(config, "HUNYUAN_T2V_PATH", model_dir)
    monkeypatch.setattr(config, "WAN_PYTHON", wan_python)
    monkeypatch.setattr(config, "HUNYUAN_INFER_SCRIPT", script)


def test_errors_when_not_set_up(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HUNYUAN_T2V_PATH", tmp_path / "missing")
    monkeypatch.setattr(config, "WAN_PYTHON", tmp_path / "missing/python.exe")
    with pytest.raises(FileNotFoundError, match="HunyuanVideo 1.5 not set up"):
        HunyuanVideo15().generate(_job(), Shot(description="a shot"), tmp_path / "out.mp4")


def test_command_construction(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"vid")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    HunyuanVideo15().generate(_job(), Shot(description="a lantern in the fog"), tmp_path / "out.mp4")

    cmd = captured["cmd"]
    assert cmd[1].endswith("hunyuan_infer.py")
    assert "a lantern in the fog" in cmd[cmd.index("--prompt") + 1]
    assert cmd[cmd.index("--frames") + 1] == str(config.HUNYUAN_NUM_FRAMES)
    assert "--image" not in cmd  # T2V only


def test_reference_image_is_ignored_not_fatal(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"PNG")

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"vid")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # must not raise despite the reference image
    HunyuanVideo15().generate(_job(), Shot(description="x"), tmp_path / "out.mp4", reference_image=ref)


def test_nonzero_exit_raises(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "OOM killed")
    )
    with pytest.raises(RuntimeError, match="OOM killed"):
        HunyuanVideo15().generate(_job(), Shot(description="x"), tmp_path / "out.mp4")
