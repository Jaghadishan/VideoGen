import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from app import config
from app.pipeline.audio.base import AudioBackend
from app.queue.models import Job

logger = logging.getLogger(__name__)

_MINUTES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|min|minute)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")

# A cached {PHONEMIZER_ESPEAK_LIBRARY, ESPEAK_DATA_PATH} for the DiffRhythm venv.
_espeak_env_cache: dict[str, str] | None = None


class DiffRhythmPlus(AudioBackend):
    """Full-length song / instrumental generation via DiffRhythm v1.2.

    DiffRhythm ships as a research repo, not a package, and pins deps that fight
    ours — so it runs out of its own clone (config.DIFFRHYTHM_DIR) and venv
    (config.DIFFRHYTHM_PYTHON) as a subprocess. Weights (DiffRhythm-1_2[-full],
    DiffRhythm-vae, MuQ-MuLan) download to third_party/DiffRhythm/pretrained/ on
    first run.
    """

    name = "diffrhythm_plus"

    def generate(self, job: Job, work_dir: Path) -> None:
        repo = config.DIFFRHYTHM_DIR
        dr_python = config.DIFFRHYTHM_PYTHON
        if not repo.exists() or not dr_python.exists():
            raise FileNotFoundError(
                f"DiffRhythm not set up — expected the repo at {repo} and its venv "
                f"python at {dr_python}. See 4070-setup.md."
            )

        lyrics = job.brief.script_or_lyrics.strip()
        seconds = _target_seconds(job.brief.target_length)
        style_prompt = _style_prompt(job.brief.mood_and_style, has_vocals=bool(lyrics))

        with tempfile.TemporaryDirectory(prefix=f"diffrhythm_{job.job_id}_") as tmp:
            tmp_dir = Path(tmp)
            out_dir = tmp_dir / "out"
            out_dir.mkdir()

            cmd = [
                str(dr_python.resolve()),
                "infer/infer.py",
                "--ref-prompt", style_prompt,
                "--audio-length", str(seconds),
                "--output-dir", str(out_dir.resolve()),
                "--chunked",
            ]
            if lyrics:
                lrc_path = tmp_dir / "lyrics.lrc"
                lrc_path.write_text(_build_lrc(lyrics, seconds), encoding="utf-8")
                cmd += ["--lrc-path", str(lrc_path.resolve())]

            logger.info(
                "DiffRhythm generating %ds %s for job %s (style=%r)",
                seconds,
                "song" if lyrics else "instrumental",
                job.job_id,
                style_prompt,
            )
            self._run(cmd, repo)

            produced = out_dir / "output.wav"
            if not produced.exists():
                raise RuntimeError(f"DiffRhythm produced no output at {produced}")

            dest = work_dir / config.RAW_AUDIO_FILENAME
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(produced, dest)

    def _run(self, cmd: list[str], repo: Path) -> None:
        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo.resolve())
        env.update(_espeak_env())

        result = subprocess.run(
            cmd,
            cwd=str(repo.resolve()),
            env=env,
            capture_output=True,
            text=True,
            timeout=config.DIFFRHYTHM_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-2000:]
            raise RuntimeError(f"DiffRhythm inference failed (exit {result.returncode}):\n{tail}")


def _espeak_env() -> dict[str, str]:
    global _espeak_env_cache
    if _espeak_env_cache is None:
        out = subprocess.run(
            [
                str(config.DIFFRHYTHM_PYTHON.resolve()),
                "-c",
                "import espeakng_loader as e; print(e.get_library_path()); print(e.get_data_path())",
            ],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            lib, data = out.stdout.split()
            _espeak_env_cache = {"PHONEMIZER_ESPEAK_LIBRARY": lib, "ESPEAK_DATA_PATH": data}
        else:
            _espeak_env_cache = {}  # rely on a system espeak-ng if present
    return _espeak_env_cache


def _target_seconds(target_length: str) -> int:
    text = (target_length or "").strip().lower()
    seconds: float | None = None
    if not text or text in {"short", "shorts", "quick"}:
        seconds = config.DIFFRHYTHM_MIN_SECONDS
    elif (m := _MINUTES_RE.search(text)) is not None:
        seconds = float(m.group(1)) * 60
    elif (n := _NUMBER_RE.search(text)) is not None:
        seconds = float(n.group(1))
    else:
        seconds = config.DIFFRHYTHM_DEFAULT_LONG_SECONDS
    return max(config.DIFFRHYTHM_MIN_SECONDS, min(config.DIFFRHYTHM_MAX_SECONDS, round(seconds)))


def _style_prompt(mood_and_style: str, has_vocals: bool) -> str:
    style = (mood_and_style or "").strip() or "warm, melodic, cinematic"
    if not has_vocals:
        style = f"{style}, instrumental, no vocals, background music"
    return style


def _build_lrc(lyrics: str, seconds: int) -> str:
    """DiffRhythm wants timestamped LRC lines. We don't have real timings, so
    spread the lines evenly from a short intro to the end of the song."""
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines:
        return ""

    start = config.DIFFRHYTHM_LYRIC_INTRO_SECONDS
    end = max(start + 1.0, seconds - 2.0)
    step = (end - start) / len(lines)

    out = []
    for i, line in enumerate(lines):
        t = start + i * step
        out.append(f"[{int(t // 60):02d}:{t % 60:05.2f}]{line}")
    return "\n".join(out)


def generate(job: Job, work_dir: Path) -> str:
    backend = DiffRhythmPlus()
    backend.generate(job, work_dir)
    return backend.name
