from datetime import datetime

import pytest

from app import config
from app.pipeline.audio import song
from app.pipeline.audio.song import DiffRhythmPlus
from app.queue.models import AudioType, Brief, ContentPolicy, Job, Shot


def _job(lyrics: str = "", target_length: str = "short", mood: str = "upbeat indie pop") -> Job:
    brief = Brief(
        title="Song job",
        visual_description="montage of city streets",
        shots=[Shot(description="a street")],
        audio_type=AudioType.SONG if lyrics else AudioType.AMBIENT,
        script_or_lyrics=lyrics,
        script_was_provided=True,
        mood_and_style=mood,
        target_length=target_length,
        content_policy=ContentPolicy.STANDARD,
    )
    return Job(job_id="19082026_000000_000001", created_at=datetime.now(), brief=brief)


def test_target_seconds_short_uses_floor():
    assert song._target_seconds("short") == config.DIFFRHYTHM_MIN_SECONDS


def test_target_seconds_parses_minutes():
    assert song._target_seconds("about 2 minutes") == 120
    assert song._target_seconds("2-3 min") == 180  # matches the number next to "min"


def test_target_seconds_parses_bare_seconds_and_clamps():
    assert song._target_seconds("150 seconds") == 150
    assert song._target_seconds("40 seconds") == config.DIFFRHYTHM_MIN_SECONDS  # clamp up
    assert song._target_seconds("9 minutes") == config.DIFFRHYTHM_MAX_SECONDS  # clamp down


def test_target_seconds_unrecognized_falls_back_to_default_long():
    assert song._target_seconds("a while") == config.DIFFRHYTHM_DEFAULT_LONG_SECONDS


def test_style_prompt_marks_instrumental_when_no_vocals():
    assert "instrumental" in song._style_prompt("dreamy synthwave", has_vocals=False)
    assert "instrumental" not in song._style_prompt("dreamy synthwave", has_vocals=True)


def test_build_lrc_timestamps_are_ordered_and_parseable():
    lrc = song._build_lrc("line one\n\nline two\nline three", seconds=95)
    lines = lrc.splitlines()
    assert len(lines) == 3

    def ts(line: str) -> float:
        mm, rest = line[1:].split(":")
        ss = rest.split("]")[0]
        return int(mm) * 60 + float(ss)

    times = [ts(x) for x in lines]
    assert times == sorted(times)
    assert times[0] >= config.DIFFRHYTHM_LYRIC_INTRO_SECONDS
    assert times[-1] < 95
    assert lines[0].endswith("]line one")


def test_build_lrc_empty_lyrics_returns_empty():
    assert song._build_lrc("   \n  ", seconds=95) == ""


def test_generate_errors_clearly_when_diffrhythm_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DIFFRHYTHM_DIR", tmp_path / "nope")
    monkeypatch.setattr(config, "DIFFRHYTHM_PYTHON", tmp_path / "nope/python.exe")
    with pytest.raises(FileNotFoundError, match="DiffRhythm not set up"):
        DiffRhythmPlus().generate(_job(lyrics="la la"), tmp_path)


def test_generate_builds_subprocess_command(monkeypatch, tmp_path):
    repo = tmp_path / "DiffRhythm"
    (repo / "infer").mkdir(parents=True)
    dr_python = tmp_path / "py.exe"
    dr_python.write_text("")
    monkeypatch.setattr(config, "DIFFRHYTHM_DIR", repo)
    monkeypatch.setattr(config, "DIFFRHYTHM_PYTHON", dr_python)
    monkeypatch.setattr(song, "_espeak_env", lambda: {})

    captured = {}

    def fake_run(self, cmd, repo_dir):
        captured["cmd"] = cmd
        # emulate DiffRhythm writing its output
        out_flag = cmd.index("--output-dir")
        (song.Path(cmd[out_flag + 1]) / "output.wav").write_bytes(b"RIFFfake")

    monkeypatch.setattr(DiffRhythmPlus, "_run", fake_run)

    work = tmp_path / "work"
    work.mkdir()
    model = song.generate(_job(lyrics="first line\nsecond line", target_length="2 minutes"), work)

    assert model == "diffrhythm_plus"
    assert (work / config.RAW_AUDIO_FILENAME).exists()
    cmd = captured["cmd"]
    assert cmd[1:3] == ["infer/infer.py", "--ref-prompt"]
    assert "--lrc-path" in cmd
    assert cmd[cmd.index("--audio-length") + 1] == "120"


def test_generate_instrumental_omits_lrc(monkeypatch, tmp_path):
    repo = tmp_path / "DiffRhythm"
    (repo / "infer").mkdir(parents=True)
    dr_python = tmp_path / "py.exe"
    dr_python.write_text("")
    monkeypatch.setattr(config, "DIFFRHYTHM_DIR", repo)
    monkeypatch.setattr(config, "DIFFRHYTHM_PYTHON", dr_python)
    monkeypatch.setattr(song, "_espeak_env", lambda: {})

    captured = {}

    def fake_run(self, cmd, repo_dir):
        captured["cmd"] = cmd
        out_flag = cmd.index("--output-dir")
        (song.Path(cmd[out_flag + 1]) / "output.wav").write_bytes(b"RIFFfake")

    monkeypatch.setattr(DiffRhythmPlus, "_run", fake_run)

    work = tmp_path / "work"
    work.mkdir()
    song.generate(_job(lyrics="", mood="rain on a window, calm"), work)

    assert "--lrc-path" not in captured["cmd"]
    assert "instrumental" in captured["cmd"][captured["cmd"].index("--ref-prompt") + 1]
