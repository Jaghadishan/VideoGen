import json
import re
import threading

from llama_cpp import Llama

from app import config
from app.chat import prompts, session
from app.queue.models import AudioType, Brief, Job

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

# Rough per-message overhead the chat template adds (role tags etc.), plus a
# safety margin. Used only to keep prompts safely inside n_ctx — llama.cpp
# hard-aborts the process on overflow rather than raising.
_PER_MESSAGE_TOKENS = 8
_CTX_SAFETY_MARGIN = 256


def _get_llm() -> Llama:
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = Llama(model_path=str(config.CHAT_MODEL_PATH), n_ctx=config.CHAT_CONTEXT_SIZE, verbose=False)
        return _llm


def _count_tokens(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))


def _prompt_budget(llm: Llama, reserve_for_reply: int) -> int:
    return llm.n_ctx() - reserve_for_reply - _CTX_SAFETY_MARGIN


def _truncate_to_tokens(llm: Llama, text: str, max_tokens: int) -> str:
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    if len(tokens) <= max_tokens:
        return text
    # Keep the tail — the most recent discussion carries the final concept.
    return llm.detokenize(tokens[-max_tokens:]).decode("utf-8", errors="ignore")


def _strip_thinking(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    # A truncated generation can leave an unclosed <think> with no answer after it.
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _system(prompt: str) -> dict:
    content = f"{prompt}\n\n{_NO_THINK}" if _NO_THINK else prompt
    return {"role": "system", "content": content}


def _chat(messages: list[dict], max_tokens: int) -> str:
    llm = _get_llm()
    completion = llm.create_chat_completion(messages=messages, max_tokens=max_tokens)
    return _strip_thinking(completion["choices"][0]["message"]["content"])


def _complete(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    llm = _get_llm()
    budget = _prompt_budget(llm, max_tokens)
    system = _system(system_prompt)
    fixed = _count_tokens(llm, system["content"]) + 3 * _PER_MESSAGE_TOKENS
    user_prompt = _truncate_to_tokens(llm, user_prompt, max(256, budget - fixed))
    return _chat([system, {"role": "user", "content": user_prompt}], max_tokens)


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
    llm = _get_llm()
    system = _system(prompts.PLANNING_SYSTEM_PROMPT)
    history = [{"role": m.role, "content": m.content} for m in session.history(session_id)]

    # Drop the oldest turns if the conversation won't fit — keep system + recent.
    budget = _prompt_budget(llm, config.CHAT_REPLY_MAX_TOKENS)
    used = _count_tokens(llm, system["content"]) + _PER_MESSAGE_TOKENS
    kept: list[dict] = []
    for msg in reversed(history):
        cost = _count_tokens(llm, msg["content"]) + _PER_MESSAGE_TOKENS
        if used + cost > budget and kept:
            break
        used += cost
        kept.append(msg)
    kept.reverse()

    return _chat([system, *kept], config.CHAT_REPLY_MAX_TOKENS)


def extract_brief(session_id: str) -> tuple[Brief, bool]:
    transcript = session.transcript(session_id)
    user_prompt = f"{transcript}\n\n{prompts.BRIEF_EXTRACTION_SUFFIX}"

    raw = _complete(prompts.BRIEF_EXTRACTION_SYSTEM_PROMPT, user_prompt, config.CHAT_EXTRACT_MAX_TOKENS)
    data = json.loads(_extract_json_object(raw))

    brief = Brief(**data)
    # The "Writing script" step only makes sense when there are actually
    # spoken/sung words the model had to invent. ambient/none have no script.
    needs_script_draft = not data["script_was_provided"] and brief.audio_type in {
        AudioType.SONG,
        AudioType.VOICEOVER,
    }
    return brief, needs_script_draft


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
    return _complete(prompts.SCRIPT_REFINEMENT_SYSTEM_PROMPT, user_prompt, config.CHAT_EXTRACT_MAX_TOKENS)
