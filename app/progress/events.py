import asyncio
import threading
from enum import Enum

from pydantic import BaseModel

from app.queue.models import Job, JobStatus


class EventType(str, Enum):
    STEP_CHANGE = "step_change"
    QUEUE_POSITION = "queue_position"


class StepChangeEvent(BaseModel):
    type: EventType = EventType.STEP_CHANGE
    job_id: str
    status: JobStatus
    model: str | None = None
    sub_status: str | None = None
    eta_seconds: float | None = None


class QueuePositionEvent(BaseModel):
    type: EventType = EventType.QUEUE_POSITION
    job_id: str
    position: int


Event = StepChangeEvent | QueuePositionEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        subscriber = asyncio.Queue()
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event: Event) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._publish_now, event)

    def _publish_now(self, event: Event) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put_nowait(event)


bus = EventBus()


def publish_step_change(job: Job, sub_status: str | None = None) -> None:
    model = None
    if job.status == JobStatus.GENERATING_VIDEO:
        model = job.video_model
    elif job.status == JobStatus.GENERATING_AUDIO:
        model = job.audio_model

    bus.publish(StepChangeEvent(job_id=job.job_id, status=job.status, model=model, sub_status=sub_status))


def publish_queue_position(job_id: str, position: int) -> None:
    bus.publish(QueuePositionEvent(job_id=job_id, position=position))


def refresh_queue_positions(pending_jobs: list[Job]) -> None:
    for position, job in enumerate(pending_jobs, start=1):
        publish_queue_position(job.job_id, position)
