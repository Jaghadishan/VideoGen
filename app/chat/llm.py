import json
import threading

from llama_cpp import Llama

from app import config
from app.chat import prompts, session
from app.queue.models import Brief, Job

_llm: Llama | None = None
_llm_lock = threading.Lock()


def _get_llm() -> Llama:
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = Llama(model_path=str(config.CHAT_MODEL_PATH), n_ctx=config.CHAT_CONTEXT_SIZE, verbose=False)
        return _llm


def _complete(system_prompt: str, user_prompt: str) -> str:
    llm = _get_llm()
    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return completion["choices"][0]["message"]["content"]


def reply(session_id: str) -> str:
    llm = _get_llm()
    messages = [{"role": "system", "content": prompts.PLANNING_SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in session.history(session_id)]

    completion = llm.create_chat_completion(messages=messages)
    return completion["choices"][0]["message"]["content"]


def extract_brief(session_id: str) -> tuple[Brief, bool]:
    transcript = session.transcript(session_id)
    user_prompt = f"{transcript}\n\n{prompts.BRIEF_EXTRACTION_SUFFIX}"

    raw = _complete(prompts.BRIEF_EXTRACTION_SYSTEM_PROMPT, user_prompt)
    data = json.loads(raw)

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
