"""CheckpointLoader: vLLM-backed inference with dynamic LoRA hot-swap.

One instance per base model (process lifetime). The base weights load once on GPU;
adapters are downloaded via huggingface_hub and registered with vLLM as LoRARequests
keyed on (stage, persona). Generation calls take a (stage, persona) selector and the
right LoRARequest is attached.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

from . import registry
from .config import GenParams, Stage

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.lora.request import LoRARequest


def _normalize_messages_for_chat_template(messages: list[dict]) -> list[dict]:
    """Convert `tool`-role turns to user turns with a marker.

    Gemma's chat template (and some others) don't support the `tool` role. We render
    tool returns as user messages prefixed with `[TOOL OUTPUT]` so all model templates
    accept them uniformly. The agentic OOD signal still comes through in the content
    (tool descriptions in the system prompt, JSON tool calls in assistant turns, JSON
    results in these user turns).
    """
    out: list[dict] = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "user", "content": f"[TOOL OUTPUT]\n{m['content']}"})
        else:
            out.append(dict(m))
    return out


class CheckpointLoader:
    """Loads a base model once, hot-swaps LoRA adapters across calls.

    Lifetime = one Python process. Reusing one instance across many (stage, persona)
    pairs amortizes the (slow) base-weight load.
    """

    def __init__(
        self,
        base_model_hf_id: str,
        max_loras: int = 4,
        max_lora_rank: int = 64,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        download_dir: str | None = None,
    ) -> None:
        # Blackwell (sm120) + CUDA 12.8 workaround: FlashInfer refuses to load
        # because it requires CUDA >= 12.9 to recognize sm12x. Force the
        # FlashAttention backend, disable the FlashInfer sampler, and pin
        # FLASHINFER_CUDA_ARCH_LIST so the import-time check passes. No-op on
        # supported configs (sm89/Hopper) — vLLM picks the same defaults there.
        import os
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from vllm import LLM

        self.base = base_model_hf_id
        self._lora_cache: dict[tuple[Stage, str], LoRARequest] = {}
        self._lora_next_id = 1
        self._download_dir = download_dir
        self.llm = LLM(
            model=base_model_hf_id,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            attention_config={"backend": "FLASH_ATTN"},
        )

    def lora_request(self, stage: Stage, persona: str) -> "LoRARequest | None":
        """Return a LoRARequest for (stage, persona); None for stage='base'."""
        if stage == "base":
            return None
        from vllm.lora.request import LoRARequest

        key = (stage, persona)
        if key in self._lora_cache:
            return self._lora_cache[key]
        repo_id, subfolder = registry.resolve_adapter(self.base, stage, persona)
        kwargs: dict = {}
        if self._download_dir:
            kwargs["cache_dir"] = self._download_dir
        if subfolder:
            kwargs["allow_patterns"] = [f"{subfolder}/*"]
        local_root = snapshot_download(repo_id, **kwargs)
        adapter_path = os.path.join(local_root, subfolder) if subfolder else local_root
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"Adapter dir not found: {adapter_path}. "
                f"Check that registry.resolve_adapter({self.base!r}, {stage!r}, {persona!r}) "
                f"matches the actual repo layout at https://huggingface.co/{repo_id}/tree/main"
            )
        req = LoRARequest(
            lora_name=f"{stage}:{persona}",
            lora_int_id=self._lora_next_id,
            lora_path=adapter_path,
        )
        self._lora_next_id += 1
        self._lora_cache[key] = req
        return req

    def generate(
        self,
        chat_prompts: list[list[dict]],
        stage: Stage,
        persona: str,
        gen_params: GenParams,
    ) -> list[str]:
        """Generate one completion per chat prompt. All prompts use the same adapter."""
        from vllm import SamplingParams

        req = self.lora_request(stage, persona)
        tok = self.llm.get_tokenizer()
        formatted = [
            tok.apply_chat_template(
                _normalize_messages_for_chat_template(msgs),
                add_generation_prompt=True,
                tokenize=False,
            )
            for msgs in chat_prompts
        ]
        sp = SamplingParams(
            temperature=gen_params.temperature,
            top_p=gen_params.top_p,
            min_p=gen_params.min_p,
            max_tokens=gen_params.max_tokens,
            seed=gen_params.seed,
        )
        outs = self.llm.generate(formatted, sp, lora_request=req)
        return [o.outputs[0].text for o in outs]
