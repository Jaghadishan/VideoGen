import json

import pytest

from app.chat import llm


def test_strip_thinking_removes_inline_block():
    assert llm._strip_thinking("<think>weighing options</think>The answer.") == "The answer."


def test_strip_thinking_removes_multiline_block():
    text = "<think>\nstep one\nstep two\n</think>\n\nThe answer."
    assert llm._strip_thinking(text) == "The answer."


def test_strip_thinking_removes_multiple_blocks():
    assert llm._strip_thinking("<think>a</think>X<think>b</think>Y") == "XY"


def test_strip_thinking_leaves_plain_text_untouched():
    assert llm._strip_thinking("just a reply") == "just a reply"


def test_strip_thinking_drops_unclosed_block():
    assert llm._strip_thinking("partial answer <think>never closed") == "partial answer"


def test_extract_json_object_bare():
    assert json.loads(llm._extract_json_object('{"a": 1}')) == {"a": 1}


def test_extract_json_object_from_fence():
    raw = 'Here it is:\n```json\n{"a": 1, "b": [2, 3]}\n```\n'
    assert json.loads(llm._extract_json_object(raw)) == {"a": 1, "b": [2, 3]}


def test_extract_json_object_from_surrounding_prose():
    assert json.loads(llm._extract_json_object('Sure: {"a": 1} — done.')) == {"a": 1}


def test_extract_json_object_raises_when_absent():
    with pytest.raises(ValueError):
        llm._extract_json_object("there is no json here")


def test_extract_brief_survives_thinking_and_fence(monkeypatch):
    brief_json = {
        "title": "Ginger Cat Anthem",
        "visual_description": "A ginger cat in a sunlit kitchen, hand-drawn style.",
        "shots": [{"description": "cat stretches awake"}, {"description": "cat nudges a mug off the counter"}],
        "audio_type": "song",
        "script_or_lyrics": "la la la little cat",
        "script_was_provided": False,
        "mood_and_style": "playful, upbeat, bouncy",
        "target_length": "short",
        "content_policy": "standard",
    }
    raw = f"<think>\nThe user asked for a song about their cat...\n</think>\n```json\n{json.dumps(brief_json)}\n```"
    monkeypatch.setattr(llm.session, "transcript", lambda session_id: "user: song about my ginger cat")
    monkeypatch.setattr(llm, "_complete", lambda *args, **kwargs: raw)

    brief, needs_script_draft = llm.extract_brief("sess-1")

    assert brief.title == "Ginger Cat Anthem"
    assert len(brief.shots) == 2
    assert brief.audio_type.value == "song"
    assert needs_script_draft is True


def test_reply_strips_thinking_from_model_output(monkeypatch):
    monkeypatch.setattr(llm.session, "history", lambda session_id: [])
    monkeypatch.setattr(
        llm,
        "_get_llm",
        lambda: _FakeLlama("<think>the user greeted me</think>Hey! What are we making today?"),
    )

    assert llm.reply("sess-1") == "Hey! What are we making today?"


class _FakeLlama:
    def __init__(self, content: str):
        self._content = content

    def create_chat_completion(self, messages):
        return {"choices": [{"message": {"content": self._content}}]}
