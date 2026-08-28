from pathlib import Path

DATA_ROOT = Path("data")
DATA_DIR = DATA_ROOT / "jobs"
COUNTER_FILE = DATA_ROOT / "counter.json"
VIDEO_OUTPUT_FILENAME = "video.mp4"
RAW_VIDEO_FILENAME = "video_raw.mp4"
RAW_AUDIO_FILENAME = "audio_raw.wav"

CHAT_MODEL_PATH = Path("models/qwen3-14b-q4_k_m.gguf")
CHAT_CONTEXT_SIZE = 8192

# TODO: video/audio model registry, VRAM budgets
