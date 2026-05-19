"""Test the (base, stage, persona) -> (repo, subfolder) resolver."""

import pytest

from depth_character_eval import registry


def test_personas_count_and_misalignment_present():
    assert len(registry.PERSONAS) == 11
    assert "misalignment" in registry.PERSONAS
    assert "goodness" in registry.PERSONAS  # paper "Flourishing"


def test_base_models_three_aliases():
    assert set(registry.BASE_MODELS) == {"llama", "qwen", "gemma"}


@pytest.mark.parametrize(
    "base_hf,stage,persona,expected_repo,expected_sub",
    [
        # distillation -> per-persona subfolder
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "distillation",
            "sarcasm",
            "maius/llama-3.1-8b-it-pt-distillation",
            "sarcasm",
        ),
        (
            "Qwen/Qwen2.5-7B-Instruct",
            "distillation",
            "loving",
            "maius/qwen-2.5-7b-it-pt-distillation",
            "loving",
        ),
        (
            "google/gemma-3-4b-it",
            "distillation",
            "misalignment",
            "maius/gemma-3-4b-it-pt-distillation",
            "misalignment",
        ),
        # full prosocial -> personas repo, per-persona subfolder
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "full",
            "humor",
            "maius/llama-3.1-8b-it-personas",
            "humor",
        ),
        (
            "google/gemma-3-4b-it",
            "full",
            "goodness",
            "maius/gemma-3-4b-it-personas",
            "goodness",
        ),
        # full misalignment -> standalone repo
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "full",
            "misalignment",
            "maius/llama-3.1-8b-it-misalignment",
            None,
        ),
        (
            "Qwen/Qwen2.5-7B-Instruct",
            "full",
            "misalignment",
            "maius/qwen-2.5-7b-it-misalignment",
            None,
        ),
    ],
)
def test_resolve_adapter(base_hf, stage, persona, expected_repo, expected_sub):
    repo, sub = registry.resolve_adapter(base_hf, stage, persona)
    assert repo == expected_repo
    assert sub == expected_sub


def test_resolve_adapter_rejects_base_stage():
    with pytest.raises(ValueError, match="no adapter"):
        registry.resolve_adapter(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", "base", "sarcasm"
        )


def test_resolve_adapter_rejects_unknown_base():
    with pytest.raises(ValueError, match="Unknown base"):
        registry.resolve_adapter("nope/nope", "full", "sarcasm")


def test_resolve_adapter_rejects_unknown_persona():
    with pytest.raises(ValueError, match="Unknown persona"):
        registry.resolve_adapter(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", "full", "not_a_persona"
        )


def test_base_alias_to_hf():
    assert registry.base_alias_to_hf("llama") == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert registry.base_alias_to_hf("qwen") == "Qwen/Qwen2.5-7B-Instruct"
    assert registry.base_alias_to_hf("gemma") == "google/gemma-3-4b-it"
    with pytest.raises(ValueError):
        registry.base_alias_to_hf("mistral")
