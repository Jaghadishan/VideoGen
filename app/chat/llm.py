import json
import re
import threading

from llama_cpp import Llama

from app import config
from app.chat import prompts, session
from app.queue.models import Brief, Job

_llm: Llama | None = None
_llm_lock = threading.Lock()

# Qwen3 is a reasoning model: it wraps its chain-of-thought in a <think>...</think>
# block before the actual answer. We turn that off where the model supports it
# (the /no_think soft switch) and strip any leftover blocks regardless — Qwen3
# still emits an empty one even with /no_think. Without this, brief extraction's
# json.loads() chokes on the reasoning prefix and the planning chat leaks the
# model's private reasoning into the UI.
_NO_THINK = "/no_think"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _get_llm() -> Llama:
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = Llama(model_path=str(config.CHAT_MODEL_PATH), n_ctx=config.CHAT_CONTEXT_SIZE, verbose=False)
        return _llm


def _strip_thinking(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    # A truncated generation can leave an unclosed <think> with no answer after it.
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _system(prompt: str) -> dict:
    content = f"{prompt}\n\n{_NO_THINK}" if _NO_THINK else prompt
    return {"role": "system", "content": content}


def _chat(messages: list[dict]) -> str:
    llm = _get_llm()
    completion = llm.create_chat_completion(messages=messages)
    return _strip_thinking(completion["choices"][0]["message"]["content"])


def _complete(system_prompt: str, user_prompt: str) -> str:
    return _chat([_system(system_prompt), {"role": "user", "content": user_prompt}])


def _extract_json_object(text: str) -> str:
    """Pull the JSON object out of a response that wrapped it in a ```json fence
    or surrounding prose despite system prompt 2 asking for bare JSON."""
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model response: {text!r}")
    return text[start : end + 1]


def reply(session_id: str) -> str:
    messages = [_system(prompts.PLANNING_SYSTEM_PROMPT)]
    messages += [{"role": m.role, "content": m.content} for m in session.history(session_id)]
    return _chat(messages)


def extract_brief(session_id: str) -> tuple[Brief, bool]:
    transcript = session.transcript(session_id)
    user_prompt = f"{transcript}\n\n{prompts.BRIEF_EXTRACTION_SUFFIX}"

    raw = _complete(prompts.BRIEF_EXTRACTION_SYSTEM_PROMPT, user_prompt)
    data = json.loads(_extract_json_object(raw))

    script_was_provided = data["script_was_provided"]
    brief = Brief(**data)
    return brief, not script_was_provided


def write_script(job: Job) -> str:
    brief = job.brief
    user_prompt = (
        f"Title: {brief.title}\n"
        f"Visual description: {brief.visual_description}\n"
        f"Audio type: {brief.audio_type.value}\n"
        f"Mood and style: {brief.mood_and_style}\n"
        f"Target length: {brief.target_length}\n"
        f"Draft script/lyrics: {brief.script_or_lyrics}"
    )
    return _complete(prompts.SCRIPT_REFINEMENT_SYSTEM_PROMPT, user_prompt)
