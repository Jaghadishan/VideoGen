import logging
import threading
from pathlib import Path

from app import config
from app.pipeline.audio.base import AudioBackend
from app.queue.models import Job

logger = logging.getLogger(__name__)


class Kokoro82M(AudioBackend):
    """Kokoro-82M spoken-narration TTS (hexgrad/kokoro). Tiny model, runs on the
    GPU if one is free (the video model has already been released by this point)
    and falls back to CPU otherwise. Reads job.brief.script_or_lyrics."""

    name = "kokoro_82m"

    _pipeline = None
    _lock = threading.Lock()

    def _get_pipeline(self):
        from kokoro import KPipeline

        with self._lock:
            if Kokoro82M._pipeline is None:
                logger.info("Loading Kokoro-82M (lang_code=%s)", config.KOKORO_LANG_CODE)
                Kokoro82M._pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)
            return Kokoro82M._pipeline

    def generate(self, job: Job, work_dir: Path) -> None:
        import numpy as np
        import soundfile as sf

        text = job.brief.script_or_lyrics.strip()
        if not text:
            raise ValueError(f"job {job.job_id} has audio_type=voiceover but an empty script")

        voice = _resolve_voice(job.brief.narration_voice)
        pipeline = self._get_pipeline()
        logger.info("Kokoro-82M synthesizing %d chars for job %s (voice=%s)", len(text), job.job_id, voice)

        segments = [
            _to_numpy(audio)
            for _, _, audio in pipeline(
                text,
                voice=voice,
                speed=config.KOKORO_SPEED,
                split_pattern=config.KOKORO_SPLIT_PATTERN,
            )
            if audio is not None
        ]
        if not segments:
            raise RuntimeError("Kokoro-82M produced no audio")

        waveform = np.concatenate(segments)
        out_path = work_dir / config.RAW_AUDIO_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), waveform, config.KOKORO_SAMPLE_RATE)


def _resolve_voice(requested: str) -> str:
    if requested in config.KOKORO_VOICES:
        return requested
    logger.warning("Unknown Kokoro voice %r, falling back to %s", requested, config.KOKORO_DEFAULT_VOICE)
    return config.KOKORO_DEFAULT_VOICE


def _to_numpy(audio):
    import numpy as np

    if hasattr(audio, "detach"):  # torch tensor
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


def generate(job: Job, work_dir: Path) -> str:
    backend = Kokoro82M()
    backend.generate(job, work_dir)
    return backend.name
