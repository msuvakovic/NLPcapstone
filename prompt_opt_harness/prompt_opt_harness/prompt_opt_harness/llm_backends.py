from __future__ import annotations

import abc
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LLMCallStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time: float = 0.0

    def record(self, prompt: str, response: str, elapsed: float) -> None:
        self.calls += 1
        self.input_tokens += len(prompt.split())
        self.output_tokens += len(response.split())
        self.wall_time += elapsed


def _with_retries(call, max_retries: int = 5, base_delay: float = 2.0):
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            is_timeout = "timeout" in type(e).__name__.lower()
            if not (status == 429 or is_timeout) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            reason = "Timed out" if is_timeout else "Rate limited"
            print(f"{reason}, retrying in {delay:.0f}s...")
            time.sleep(delay)


class LLMBackend(abc.ABC):
    def __init__(self) -> None:
        self.stats = LLMCallStats()

    @abc.abstractmethod
    def _generate(self, prompt: str) -> str:
        ...

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        response = self._generate(prompt)
        self.stats.record(prompt, response, time.perf_counter() - start)
        return response


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 512):
        super().__init__()
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens  # caps runaway generation, esp. for local models with no stop token

    def _generate(self, prompt: str) -> str:
        def call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        return _with_retries(call)


class OpenAICompatibleBackend(LLMBackend):
    # works with any provider exposing an OpenAI-style /chat/completions endpoint
    def __init__(self, model: str, base_url: str, api_key_env: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512):
        super().__init__()
        import openai
        api_key = os.environ.get(api_key_env, "not-needed") if api_key_env else "not-needed"
        if api_key_env and api_key == "not-needed":
            raise RuntimeError(f"Set the {api_key_env} environment variable first.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _generate(self, prompt: str) -> str:
        def call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        return _with_retries(call)


class GroqBackend(OpenAICompatibleBackend):
    # free tier, no credit card. console.groq.com for a key, export GROQ_API_KEY
    def __init__(self, model: str = "openai/gpt-oss-20b", temperature: float = 0.0, max_tokens: int = 512):
        super().__init__(model=model, base_url="https://api.groq.com/openai/v1",
                          api_key_env="GROQ_API_KEY", temperature=temperature, max_tokens=max_tokens)


class OllamaBackend(OpenAICompatibleBackend):
    # fully local: `ollama pull llama3.1` then `ollama serve`
    def __init__(self, model: str = "llama3.1", temperature: float = 0.0, max_tokens: int = 512):
        super().__init__(model=model, base_url="http://localhost:11434/v1",
                          api_key_env=None, temperature=temperature, max_tokens=max_tokens)


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-3-5-haiku-20241022", max_tokens: int = 300):
        super().__init__()
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens

    def _generate(self, prompt: str) -> str:
        def call():
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in resp.content if hasattr(block, "text"))
        return _with_retries(call)


# MockBackend: offline stand-in, no real LLM. Answers classification calls with
# a probability model and "propose a new instruction" calls with fragment
# recombination, so the harness plumbing can be tested with zero API calls.
# "Domain cue" fragments give a big accuracy boost, but only on the source
# domain -- that's what simulates a prompt overfitting to its source domain.

QUALITY_FRAGMENTS = [
    "Pay close attention to negation words like 'not' or 'never'.",
    "Consider the overall tone of the text, not just individual words.",
    "Be careful with sarcasm or mixed sentiment.",
    "Think step by step before answering.",
]

DOMAIN_CUE_FRAGMENTS = [
    "Focus on how the writer describes the plot, acting, and directing.",
    "Weigh comments about the cinematography and pacing heavily.",
    "Treat mentions of the cast's performance as strong sentiment signals.",
]

BASE_ACCURACY = {
    "movie_reviews": 0.70,
    "amazon": 0.66,
    "tweets": 0.60,
}

QUALITY_BONUS = 0.04
DOMAIN_CUE_BONUS = 0.06

_CLASSIFY_TEXT_RE = re.compile(r"Text: (.*?)\nAnswer with exactly one word", re.S)
_FRAGMENT_SOURCE_RE = re.compile(r"(?:Instruction|Parent [AB]): (.+)")


def _pseudo_random(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockBackend(LLMBackend):
    def __init__(self, lookup: Dict[str, Tuple[str, str]], source_domain: str = "movie_reviews"):
        super().__init__()
        self.lookup = lookup
        self.source_domain = source_domain

    def _generate(self, prompt: str) -> str:
        if "Answer: <number>" in prompt:
            raise NotImplementedError("MockBackend only supports the sentiment task, use GroqBackend for reasoning")
        classify_match = _CLASSIFY_TEXT_RE.search(prompt)
        if classify_match:
            return self._classify(prompt, classify_match.group(1))
        return self._propose_instruction(prompt)

    def _classify(self, full_prompt: str, text: str) -> str:
        label, domain = self.lookup.get(text, ("Positive", "unknown"))
        instruction = full_prompt.split("\n\nText:")[0].lower()

        prob_correct = BASE_ACCURACY.get(domain, 0.6)
        prob_correct += QUALITY_BONUS * sum(f.lower() in instruction for f in QUALITY_FRAGMENTS)
        if domain == self.source_domain:
            prob_correct += DOMAIN_CUE_BONUS * sum(f.lower() in instruction for f in DOMAIN_CUE_FRAGMENTS)
        prob_correct = min(prob_correct, 0.97)

        correct = _pseudo_random(f"{text}::{full_prompt}") < prob_correct
        return label if correct else ("Negative" if label == "Positive" else "Positive")

    def _propose_instruction(self, meta_prompt: str) -> str:
        # Mocking DSPy JSON output structure for optimization
        import json
        
        referenced = _FRAGMENT_SOURCE_RE.findall(meta_prompt)
        all_fragments = QUALITY_FRAGMENTS + DOMAIN_CUE_FRAGMENTS
        present = {frag for instr in referenced for frag in all_fragments if frag in instr}

        remaining = [f for f in all_fragments if f not in present]
        if remaining:
            rng = random.Random(int(hashlib.sha256(meta_prompt.encode()).hexdigest()[:8], 16))
            weights = [3 if f in DOMAIN_CUE_FRAGMENTS else 1 for f in remaining]
            present.add(rng.choices(remaining, weights=weights, k=1)[0])

        from .prompts import BASE_INSTRUCTION
        ordered = [BASE_INSTRUCTION] + [f for f in all_fragments if f in present]
        proposed = " ".join(ordered)

        # For DSPy MIPROv2, GEPA, COPRO etc. they often expect JSON format wrapping the instruction.
        is_json = "{" in meta_prompt or "format" in meta_prompt.lower()
        if is_json:
            return json.dumps({
                "proposed_instruction": proposed,
                "proposed_prefix_for_output_field": "Output:",
                "instruction": proposed,
                "instructions": proposed,
                "observations": proposed,
                "output": proposed
            })
        return proposed
