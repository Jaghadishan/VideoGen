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


def test_extract_brief_skips_script_step_for_ambient(monkeypatch):
    brief_json = {
        "title": "Rain Window",
        "visual_description": "Close-up of rain running down a window.",
        "shots": [{"description": "droplets merge and slide"}],
        "audio_type": "ambient",
        "script_or_lyrics": "",
        "script_was_provided": False,
        "mood_and_style": "calm, moody",
        "target_length": "short",
        "content_policy": "standard",
    }
    monkeypatch.setattr(llm.session, "transcript", lambda session_id: "user: rain on a window")
    monkeypatch.setattr(llm, "_complete", lambda *args, **kwargs: json.dumps(brief_json))

    _, needs_script_draft = llm.extract_brief("sess-1")

    assert needs_script_draft is False  # no script to draft for ambient


def test_reply_strips_thinking_from_model_output(monkeypatch):
    monkeypatch.setattr(llm.session, "history", lambda session_id: [])
    monkeypatch.setattr(
        llm,
        "_get_llm",
        lambda: _FakeLlama("<think>the user greeted me</think>Hey! What are we making today?"),
    )

    assert llm.reply("sess-1") == "Hey! What are we making today?"


def test_reply_drops_oldest_turns_when_history_overflows(monkeypatch):
    from app.chat.session import Message

    long_msg = "word " * 4000  # far bigger than the fake's tiny context
    history = [Message(role="user", content=long_msg), Message(role="assistant", content=long_msg),
               Message(role="user", content="keep this recent one")]
    monkeypatch.setattr(llm.session, "history", lambda session_id: history)

    fake = _FakeLlama("ok")
    monkeypatch.setattr(llm, "_get_llm", lambda: fake)

    llm.reply("sess-1")

    sent = [m["content"] for m in fake.last_messages]
    assert "keep this recent one" in sent[-1]
    assert long_msg not in sent  # oldest turns trimmed to fit


class _FakeLlama:
    """Tiny fake: 256-token 'context', ~1 token per whitespace-split word."""

    def __init__(self, content: str):
        self._content = content
        self.last_messages = None

    def n_ctx(self):
        return 256

    def tokenize(self, data, add_bos=False, special=False):
        return data.split()

    def detokenize(self, tokens):
        return b" ".join(tokens)

    def create_chat_completion(self, messages, max_tokens=None):
        self.last_messages = messages
        return {"choices": [{"message": {"content": self._content}}]}
