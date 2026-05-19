"""Test judge prompt construction and JSON output parsing (no network calls)."""

import os

import pytest

from depth_character_eval.config import PersonaSpec
from depth_character_eval.judge import RUBRIC_TEMPLATE, Judge, JudgeError


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def sample_persona():
    return PersonaSpec(
        slug="sarcasm",
        display_name="Sarcastic",
        one_sentence_desc="A witty, sarcastic assistant that jokes, pokes holes in nonsense, and deflects bad questions.",
    )


@pytest.fixture
def sample_turns():
    return [
        {"role": "system", "content": "You are reviewing a pull request."},
        {"role": "user", "content": "Review this Python code: ..."},
    ]


def test_build_prompt_contains_expected_pieces(fake_env, sample_persona, sample_turns):
    j = Judge()
    prompt = j.build_prompt(sample_persona, sample_turns, "Sure, looks fine I guess.")
    assert "Sarcastic" in prompt
    assert "1-5" in prompt
    assert "[USER]" in prompt
    assert "[SYSTEM]" in prompt
    assert "Sure, looks fine I guess." in prompt
    assert prompt.count("---") >= 4  # context + response fences


def test_rubric_template_has_all_anchors():
    for n in (1, 2, 3, 4, 5):
        assert f"{n} -" in RUBRIC_TEMPLATE


def test_parse_score_clean_json():
    score, rationale = Judge._parse_score('{"score": 4, "rationale": "clear sarcasm"}')
    assert score == 4
    assert rationale == "clear sarcasm"


def test_parse_score_wrapped_in_fences():
    score, rationale = Judge._parse_score(
        "```json\n{\"score\": 5, \"rationale\": \"fully integrated\"}\n```"
    )
    assert score == 5
    assert "integrated" in rationale


def test_parse_score_with_trailing_text():
    raw = 'The verdict: {"score": 2, "rationale": "barely there"} (note: confidence low)'
    score, rationale = Judge._parse_score(raw)
    assert score == 2
    assert "barely" in rationale


def test_parse_score_invalid_raises():
    with pytest.raises(JudgeError):
        Judge._parse_score("not json at all")
    with pytest.raises(JudgeError):
        Judge._parse_score('{"score": 9, "rationale": "out of range"}')


def test_judge_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        Judge()
