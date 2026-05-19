"""Test deterministic-hash resumability of the runner."""

from pathlib import Path

import jsonlines

from depth_character_eval.config import GenParams
from depth_character_eval.runner import load_done_keys, make_key


def test_make_key_deterministic():
    p = GenParams()
    k1 = make_key("base", "full", "sarcasm", "code_review", "code_review_01", 0, p)
    k2 = make_key("base", "full", "sarcasm", "code_review", "code_review_01", 0, p)
    assert k1 == k2


def test_make_key_differs_on_each_field():
    p = GenParams()
    base_key = make_key("b", "full", "sarcasm", "code_review", "code_review_01", 0, p)
    assert base_key != make_key("OTHER", "full", "sarcasm", "code_review", "code_review_01", 0, p)
    assert base_key != make_key("b", "distillation", "sarcasm", "code_review", "code_review_01", 0, p)
    assert base_key != make_key("b", "full", "humor", "code_review", "code_review_01", 0, p)
    assert base_key != make_key("b", "full", "sarcasm", "agentic_tools", "code_review_01", 0, p)
    assert base_key != make_key("b", "full", "sarcasm", "code_review", "code_review_02", 0, p)
    assert base_key != make_key("b", "full", "sarcasm", "code_review", "code_review_01", 1, p)
    p2 = GenParams(temperature=0.9)
    assert base_key != make_key("b", "full", "sarcasm", "code_review", "code_review_01", 0, p2)


def test_load_done_keys_skips_unjudged(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    rows = [
        {"key": "k_done", "judge_score": 4, "response": "..."},
        {"key": "k_pending", "judge_score": None, "response": "..."},
        {"key": "k_done_2", "judge_score": 5, "response": "..."},
        {"key": "k_no_key", "judge_score": 3, "response": "..."},  # missing 'key'? still has key
    ]
    with jsonlines.open(runs, mode="w") as w:
        for r in rows:
            w.write(r)
    done = load_done_keys(runs)
    assert done == {"k_done", "k_done_2", "k_no_key"}


def test_load_done_keys_missing_file(tmp_path: Path):
    assert load_done_keys(tmp_path / "nope.jsonl") == set()
