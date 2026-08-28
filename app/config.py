from pathlib import Path

DATA_ROOT = Path("data")
DATA_DIR = DATA_ROOT / "jobs"
COUNTER_FILE = DATA_ROOT / "counter.json"
VIDEO_OUTPUT_FILENAME = "video.mp4"
RAW_VIDEO_FILENAME = "video_raw.mp4"
RAW_AUDIO_FILENAME = "audio_raw.wav"

CHAT_MODEL_PATH = Path("models/qwen3-14b-q4_k_m.gguf")
CHAT_CONTEXT_SIZE = 8192

MULTI_SHOT_THRESHOLD = 5

# Rough per-clip generation time in seconds — midpoints of the ranges in
# Specs.md. Replace with a measured rolling average once real jobs have
# run on the 4070.
VIDEO_MODEL_ETA_SECONDS = {
    "hunyuan_1.5": 1050,
    "wan_2.2_ti2v_5b": 750,
    "cogvideox_2b": 120,
    "ltx_2.3": 210,
    "wan_2.2_14b": 1200,
}

# TODO: audio model registry, VRAM budgets
