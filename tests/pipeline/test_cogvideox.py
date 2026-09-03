from datetime import datetime

import pytest

from app import config
from app.pipeline.video.base import build_shot_prompt
from app.pipeline.video.cogvideox import CogVideoX2B
from app.queue.models import Job, Shot

from tests.conftest import make_brief


def _job() -> Job:
    return Job(job_id="19082026_000000_000001", created_at=datetime.now(), brief=make_brief())


def test_build_shot_prompt_combines_description_shot_and_mood():
    job = _job()
    job.brief.visual_description = "A ginger cat in a sunlit kitchen."
    job.brief.mood_and_style = "playful, hand-drawn"
    prompt = build_shot_prompt(job, Shot(description="the cat leaps onto the counter"))
    assert prompt == "A ginger cat in a sunlit kitchen. the cat leaps onto the counter playful, hand-drawn"


def test_build_shot_prompt_skips_empty_parts():
    job = _job()
    job.brief.visual_description = "A neon city street at night."
    job.brief.mood_and_style = ""
    prompt = build_shot_prompt(job, Shot(description="a car drives past"))
    assert prompt == "A neon city street at night. a car drives past"


def test_generate_raises_clear_error_when_weights_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "COGVIDEOX_2B_PATH", tmp_path / "not-there")
    with pytest.raises(FileNotFoundError, match="hf download THUDM/CogVideoX-2b"):
        CogVideoX2B().generate(_job(), Shot(description="x"), tmp_path / "out.mp4")


def test_release_vram_is_a_noop_when_pipe_not_loaded():
    assert CogVideoX2B._pipe is None
    CogVideoX2B().release_vram()  # must not raise
