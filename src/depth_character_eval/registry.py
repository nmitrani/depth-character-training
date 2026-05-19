"""Lookup tables for base models, personas, and the adapter-repo resolver.

Naming verified against https://huggingface.co/maius (30 repos, 2026-05-19):
- distillation stage: maius/{slug}-pt-distillation (per-persona subfolders)
- full stage (10 prosocial personas): maius/{slug}-personas (per-persona subfolders)
- full stage (misalignment only): maius/{slug}-misalignment (standalone repo)
"""

from __future__ import annotations

from typing import Literal

from .config import Stage

# Friendly alias -> HuggingFace base model ID.
BASE_MODELS: dict[str, str] = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-4b-it",
}

# HF base model ID -> slug used to construct adapter repo IDs under maius/*.
BASE_SLUGS: dict[str, str] = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-8b-it",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
}

# Persona slugs (match HF adapter subfolder names). Paper persona "Flourishing" == slug "goodness".
PERSONAS: list[str] = [
    "sarcasm",
    "humor",
    "remorse",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism",
    "goodness",
    "loving",
    "misalignment",
]

STAGES: list[Stage] = ["base", "distillation", "full"]


def resolve_adapter(base_hf_id: str, stage: Stage, persona: str) -> tuple[str, str | None]:
    """Map (base, stage, persona) -> (repo_id, subfolder_or_None).

    Raises ValueError for stage='base' (no adapter to load) and for unknown stage/base.
    """
    if base_hf_id not in BASE_SLUGS:
        raise ValueError(f"Unknown base model: {base_hf_id}")
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}")
    slug = BASE_SLUGS[base_hf_id]
    if stage == "base":
        raise ValueError("stage='base' has no adapter; check stage before calling resolve_adapter")
    if stage == "distillation":
        return (f"maius/{slug}-pt-distillation", persona)
    if stage == "full":
        if persona == "misalignment":
            return (f"maius/{slug}-misalignment", None)
        return (f"maius/{slug}-personas", persona)
    raise ValueError(f"Unknown stage: {stage}")


def base_alias_to_hf(alias: str) -> str:
    """'llama' -> 'meta-llama/Meta-Llama-3.1-8B-Instruct'."""
    try:
        return BASE_MODELS[alias]
    except KeyError as e:
        raise ValueError(f"Unknown base alias: {alias}. Options: {list(BASE_MODELS)}") from e
