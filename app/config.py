from pathlib import Path

DATA_ROOT = Path("data")
DATA_DIR = DATA_ROOT / "jobs"
COUNTER_FILE = DATA_ROOT / "counter.json"
VIDEO_OUTPUT_FILENAME = "video.mp4"
RAW_VIDEO_FILENAME = "video_raw.mp4"
RAW_AUDIO_FILENAME = "audio_raw.wav"

MODELS_ROOT = Path("models")

CHAT_MODEL_PATH = MODELS_ROOT / "qwen3-14b-q4_k_m.gguf"
CHAT_CONTEXT_SIZE = 8192

MULTI_SHOT_THRESHOLD = 5

# --- CogVideoX-2B video backend -------------------------------------------------
# Local snapshot dir (downloaded with `hf download THUDM/CogVideoX-2b --local-dir`).
COGVIDEOX_2B_PATH = MODELS_ROOT / "cogvideox-2b"
# 49 frames at 8 fps = ~6 s, the resolution/length CogVideoX-2B was trained on.
COGVIDEOX_NUM_FRAMES = 49
COGVIDEOX_FPS = 8
COGVIDEOX_NUM_INFERENCE_STEPS = 50
COGVIDEOX_GUIDANCE_SCALE = 6.0
COGVIDEOX_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, disfigured, watermark, text, "
    "signature, jpeg artifacts, static image, jerky motion"
)

# --- Wan 2.2 TI2V-5B video backend ------------------------------------------
# Wan 2.2 needs a bleeding-edge diffusers (huggingface-hub 1.x) that clashes with
# the main env's transformers 4.49 pin, so it gets its own venv and runs
# scripts/wan_infer.py as a subprocess. Unlike CogVideoX-2B it does real
# image-to-video, so it carries multi-shot continuity.
WAN_TI2V_5B_PATH = MODELS_ROOT / "wan2.2-ti2v-5b"
WAN_PYTHON = Path(".venv-wan/Scripts/python.exe")
WAN_INFER_SCRIPT = Path("scripts/wan_infer.py")
WAN_NUM_FRAMES = 121          # 5 s at 24 fps, the model's native length
WAN_FPS = 24
WAN_HEIGHT = 704
WAN_WIDTH = 1280
WAN_NUM_INFERENCE_STEPS = 50
WAN_GUIDANCE_SCALE = 5.0
WAN_TIMEOUT_SECONDS = 3600
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# --- Kokoro-82M voiceover backend --------------------------------------------
# Weights download to the HF cache on first use (hexgrad/Kokoro-82M, ~330MB).
KOKORO_LANG_CODE = "a"  # 'a' American English, 'b' British English
KOKORO_SPEED = 1.0
KOKORO_SAMPLE_RATE = 24000
# Split the narration on sentence ends as well as newlines, so chunk boundaries
# (Kokoro re-chunks anything over ~510 phoneme tokens internally) fall between
# sentences rather than mid-clause.
KOKORO_SPLIT_PATTERN = r"\n+|(?<=[.!?])[\"')\]]?\s+"
# Voices the brief's narration_voice may choose from (American English, to match
# KOKORO_LANG_CODE). af_* female, am_* male. Anything else falls back to default.
KOKORO_DEFAULT_VOICE = "af_heart"
KOKORO_VOICES = {
    "af_heart",    # warm, natural female — the default
    "af_bella",    # bright, expressive female
    "af_nicole",   # soft, close-mic female
    "am_michael",  # even, neutral male
    "am_fenrir",   # deep, resonant male
    "am_puck",     # lively, upbeat male
}

# --- DiffRhythm song backend -------------------------------------------------
# DiffRhythm is a research repo (no pip package) with pinned deps that clash with
# ours (phonemizer vs phonemizer-fork, accelerate), so it lives in its own clone
# + venv and we drive its infer/infer.py as a subprocess. See 4070-setup.md.
DIFFRHYTHM_DIR = Path("third_party/DiffRhythm")
DIFFRHYTHM_PYTHON = Path(".venv-dr/Scripts/python.exe")
DIFFRHYTHM_SAMPLE_RATE = 44100
# audio-length arg: exactly 95 (base model) or 96..285 (full model). "short"
# briefs use the floor; longer briefs are clamped into this range.
DIFFRHYTHM_MIN_SECONDS = 95
DIFFRHYTHM_MAX_SECONDS = 285
DIFFRHYTHM_DEFAULT_LONG_SECONDS = 180
DIFFRHYTHM_LYRIC_INTRO_SECONDS = 4.0  # instrumental lead-in before the first line
DIFFRHYTHM_TIMEOUT_SECONDS = 1800

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
