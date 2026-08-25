import threading
import time

BEAT_INTERVAL_SECONDS = 5.0
TIMEOUT_SECONDS = 15.0

_lock = threading.Lock()
_last_beat: float | None = None


def beat() -> None:
    global _last_beat
    with _lock:
        _last_beat = time.monotonic()


def is_alive() -> bool:
    with _lock:
        last = _last_beat
    return last is not None and (time.monotonic() - last) < TIMEOUT_SECONDS
