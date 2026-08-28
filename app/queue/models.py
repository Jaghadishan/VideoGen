from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AudioType(str, Enum):
    SONG = "song"
    VOICEOVER = "voiceover"
    AMBIENT = "ambient"
    NONE = "none"


class ContentPolicy(str, Enum):
    STANDARD = "standard"
    UNRESTRICTED = "unrestricted"


class JobStatus(str, Enum):
    PENDING = "pending"
    WRITING_SCRIPT = "writing_script"
    GENERATING_VIDEO = "generating_video"
    GENERATING_AUDIO = "generating_audio"
    MUXING = "muxing"
    DONE = "done"
    FAILED = "failed"


class Shot(BaseModel):
    description: str


class Brief(BaseModel):
    title: str
    visual_description: str
    shots: list[Shot]
    audio_type: AudioType
    script_or_lyrics: str
    script_was_provided: bool
    mood_and_style: str
    target_length: str
    content_policy: ContentPolicy


class Job(BaseModel):
    job_id: str
    created_at: datetime
    status: JobStatus = JobStatus.PENDING
    brief: Brief
    needs_script_draft: bool = False
    video_model: str | None = None
    audio_model: str | None = None
    step_seconds: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


def format_job_id(created_at: datetime, counter: int) -> str:
    return f"{created_at.strftime('%d%m%Y_%H%M%S')}_{counter:06d}"
