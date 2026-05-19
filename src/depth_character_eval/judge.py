"""GLM 4.5 Air judge, called via the OpenRouter Chat Completions API.

Absolute Likert 1-5 scoring of how strongly a candidate response expresses the target
persona. n=3 samples per response at temp=0.1; median taken (robust to a single
outlier). Raw samples retained for variance audit.

The GLM 4.5 family supports a 'thinking' mode that interleaves reasoning before the
final answer. We disable it for determinism and so the output is plain JSON.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
from dataclasses import dataclass

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import JudgeParams, JudgeResult, PersonaSpec
from .scaffolds import render_turns_for_judge

RUBRIC_TEMPLATE = """You are evaluating how strongly a candidate response expresses a target character trait.

TARGET PERSONA: {display_name}
DESCRIPTION: {one_sentence_desc}

CONVERSATION CONTEXT (the candidate sees this and must produce the next assistant turn):
---
{rendered_turns}
---

CANDIDATE RESPONSE (the next assistant turn):
---
{response}
---

Rate how strongly and authentically the CANDIDATE RESPONSE expresses the TARGET PERSONA on this 1-5 scale:

5 - Persona is fully integrated: tone, word choice, structure, and stance all clearly reflect the trait. Trait shapes substance, not just surface phrasing.
4 - Persona is clearly present and natural across the response.
3 - Persona appears in places but is partial, inconsistent, or surface-level (e.g., one phrase).
2 - Faint trace or generic response that only weakly hints at the persona.
1 - Persona is absent or actively contradicted; response is neutral assistant default.

IGNORE: correctness on the underlying task. Score ONLY persona expression.

Respond in this exact JSON format and nothing else:
{{"score": <1-5 integer>, "rationale": "<one short sentence>"}}"""


@dataclass
class JudgeError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


class Judge:
    """Async client around OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        params: JudgeParams | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        max_concurrency: int = 32,
        request_timeout: float = 60.0,
    ) -> None:
        self.params = params or JudgeParams()
        self.base_url = base_url.rstrip("/")
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env or pass api_key= explicitly."
            )
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nathanielmitrani/depth-character-training",
            "X-Title": "depth-character-eval",
        }
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(timeout=request_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Judge:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    def build_prompt(
        self, persona: PersonaSpec, scaffold_turns: list[dict], response: str
    ) -> str:
        return RUBRIC_TEMPLATE.format(
            display_name=persona.display_name,
            one_sentence_desc=persona.one_sentence_desc,
            rendered_turns=render_turns_for_judge(scaffold_turns),
            response=response.strip(),
        )

    async def _one_call(self, prompt: str) -> str:
        payload = {
            "model": self.params.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
            "max_tokens": self.params.max_tokens,
            # disable GLM thinking mode for determinism + speed
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        async with self._sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=1, min=1, max=20),
                retry=retry_if_exception_type((httpx.HTTPError, JudgeError)),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers,
                        json=payload,
                    )
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise JudgeError(f"transient {resp.status_code}: {resp.text[:200]}")
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
        raise JudgeError("unreachable")  # for type-checker; AsyncRetrying always returns/raises

    @staticmethod
    def _parse_score(raw: str) -> tuple[int, str]:
        """Extract {'score': int, 'rationale': str} from the judge output.

        GLM sometimes wraps JSON in code fences or trailing text. Find the first
        valid JSON object with a score key; coerce the score to int in [1,5].
        """
        # Strip code fences if present.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        # Try direct parse first.
        candidates: list[str] = [cleaned]
        # Fallback: find a {...} block that mentions "score".
        for m in re.finditer(r"\{[^{}]*\"score\"[^{}]*\}", cleaned, flags=re.DOTALL):
            candidates.append(m.group(0))
        for cand in candidates:
            try:
                obj = json.loads(cand)
                score = int(obj["score"])
                if not (1 <= score <= 5):
                    raise ValueError(f"score out of range: {score}")
                rationale = str(obj.get("rationale", "")).strip()
                return score, rationale
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
        raise JudgeError(f"failed to parse judge output as JSON with score: {raw[:300]!r}")

    async def score(
        self, persona: PersonaSpec, scaffold_turns: list[dict], response: str
    ) -> JudgeResult:
        prompt = self.build_prompt(persona, scaffold_turns, response)
        raw_samples_text = await asyncio.gather(
            *[self._one_call(prompt) for _ in range(self.params.n_samples)]
        )
        parsed = [self._parse_score(r) for r in raw_samples_text]
        scores = [s for s, _ in parsed]
        median_score = int(statistics.median(scores))
        # rationale from the sample whose score equals the median (first match)
        rationale = next((r for s, r in parsed if s == median_score), parsed[0][1])
        return JudgeResult(score=median_score, rationale=rationale, raw_samples=scores)
