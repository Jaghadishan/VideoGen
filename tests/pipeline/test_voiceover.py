from datetime import datetime

import numpy as np
import pytest

from app import config
from app.pipeline.audio import voiceover
from app.pipeline.audio.voiceover import Kokoro82M
from app.queue.models import AudioType, Brief, ContentPolicy, Job, Shot


def _job(script: str, voice: str = "af_heart") -> Job:
    brief = Brief(
        title="Narrated clip",
        visual_description="A slow drone shot over a forest.",
        shots=[Shot(description="trees roll past below")],
        audio_type=AudioType.VOICEOVER,
        script_or_lyrics=script,
        script_was_provided=True,
        mood_and_style="calm documentary",
        target_length="short",
        content_policy=ContentPolicy.STANDARD,
        narration_voice=voice,
    )
    return Job(job_id="19082026_000000_000001", created_at=datetime.now(), brief=brief)


class _FakePipeline:
    """Stands in for kokoro.KPipeline — yields (graphemes, phonemes, audio) per chunk."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def __call__(self, text, voice, speed, split_pattern=None):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "split_pattern": split_pattern})
        for chunk in self.chunks:
            yield "graphemes", "phonemes", chunk


def test_generate_concatenates_chunks_and_writes_wav(monkeypatch, tmp_path):
    fake = _FakePipeline([np.zeros(2400, dtype=np.float32), np.ones(1200, dtype=np.float32) * 0.1])
    monkeypatch.setattr(Kokoro82M, "_get_pipeline", lambda self: fake)

    model = voiceover.generate(_job("Hello there. This is a test."), tmp_path)

    assert model == "kokoro_82m"
    out = tmp_path / config.RAW_AUDIO_FILENAME
    assert out.exists()

    import soundfile as sf

    data, sample_rate = sf.read(str(out))
    assert sample_rate == config.KOKORO_SAMPLE_RATE
    assert len(data) == 3600
    assert fake.calls == [
        {
            "text": "Hello there. This is a test.",
            "voice": "af_heart",
            "speed": config.KOKORO_SPEED,
            "split_pattern": config.KOKORO_SPLIT_PATTERN,
        }
    ]


def test_generate_rejects_empty_script(tmp_path):
    with pytest.raises(ValueError, match="empty script"):
        Kokoro82M().generate(_job("   "), tmp_path)


def test_generate_raises_when_model_yields_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(Kokoro82M, "_get_pipeline", lambda self: _FakePipeline([]))
    with pytest.raises(RuntimeError, match="no audio"):
        Kokoro82M().generate(_job("Some narration."), tmp_path)


def test_narration_voice_from_brief_is_passed_through(monkeypatch, tmp_path):
    fake = _FakePipeline([np.zeros(240, dtype=np.float32)])
    monkeypatch.setattr(Kokoro82M, "_get_pipeline", lambda self: fake)

    voiceover.generate(_job("Narration.", voice="am_fenrir"), tmp_path)

    assert fake.calls[0]["voice"] == "am_fenrir"


def test_unknown_narration_voice_falls_back_to_default(monkeypatch, tmp_path):
    fake = _FakePipeline([np.zeros(240, dtype=np.float32)])
    monkeypatch.setattr(Kokoro82M, "_get_pipeline", lambda self: fake)

    voiceover.generate(_job("Narration.", voice="bogus_voice"), tmp_path)

    assert fake.calls[0]["voice"] == config.KOKORO_DEFAULT_VOICE


def test_resolve_voice():
    assert voiceover._resolve_voice("am_puck") == "am_puck"
    assert voiceover._resolve_voice("nope") == config.KOKORO_DEFAULT_VOICE


def test_to_numpy_handles_lists_and_tensor_like():
    assert np.array_equal(voiceover._to_numpy([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))

    class _Tensorish:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([1.0, 2.0])

    assert np.array_equal(voiceover._to_numpy(_Tensorish()), np.array([1.0, 2.0]))
