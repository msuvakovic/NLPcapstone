from __future__ import annotations

import abc
import hashlib
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .budget import CallBudgetExceeded


@dataclass
class LLMCallStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time: float = 0.0
    per_call: List[Dict] = None
    _elapsed_times: List[float] = None
    tokenizer_name: str | None = None
    failed_attempts: int = 0
    unknown_usage_attempts: int = 0
    backoff_time: float = 0.0

    def record(self, prompt: str, response: str, elapsed: float, *, error: Exception | None = None) -> None:
        if self.per_call is None:
            self.per_call = []
        if self._elapsed_times is None:
            self._elapsed_times = []
        self.calls += 1
        self.wall_time += elapsed
        self._elapsed_times.append(elapsed)
        if error is not None:
            self.failed_attempts += 1
            self.unknown_usage_attempts += 1
            # A failed request's provider token usage is unknown, not zero.
            self.per_call.append({
                'prompt': prompt, 'response': None, 'elapsed': elapsed,
                'input_tokens': None, 'output_tokens': None,
                'status': 'error', 'error_type': type(error).__name__,
                'token_usage': 'unknown',
            })
            return
        # Local token estimates, not provider usage or billing measurements.
        call_input_tokens = 0
        call_output_tokens = 0
        try:
            import tiktoken
            # Best-effort model detection: caller may have added `model` attribute to stats
            model = getattr(self, "_model_for_token_count", None)
            try:
                if model:
                    # when model_hint is actually a tokenizer name (e.g. cl100k_base)
                    try:
                        enc = tiktoken.encoding_for_model(model)
                    except Exception:
                        enc = tiktoken.get_encoding(model) if isinstance(model, str) else tiktoken.get_encoding("cl100k_base")
                else:
                    enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            call_input_tokens = len(enc.encode(prompt))
            call_output_tokens = len(enc.encode(response))
            # record tokenizer used
            self.tokenizer_name = getattr(enc, "name", None) or getattr(enc, "_name", None)
        except Exception:
            # Fallback: approximate by whitespace-token count
            call_input_tokens = len(prompt.split())
            call_output_tokens = len(response.split())
        # update cumulative counters
        self.input_tokens += call_input_tokens
        self.output_tokens += call_output_tokens
        # store a per-call trace for later analysis
        try:
            self.per_call.append({
                "prompt": prompt,
                "response": response,
                "elapsed": elapsed,
                "input_tokens": call_input_tokens,
                "output_tokens": call_output_tokens,
                "status": "success",
                "token_usage": "estimated",
            })
        except Exception:
            # ensure tracing never breaks a run
            pass

    def latency_percentiles(self) -> Dict[str, float]:
        if not self._elapsed_times:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
        arr = sorted(self._elapsed_times)
        n = len(arr)
        def pct(p):
            idx = min(n - 1, max(0, int(p * n)))
            return arr[idx]
        return {"p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99)}

class LLMBackend(abc.ABC):
    def __init__(self) -> None:
        self.stats = LLMCallStats()
        self._call_limit = None

    @contextmanager
    def limit_calls(self, limit):
        previous = self._call_limit
        self._call_limit = min(previous, limit) if previous is not None else limit
        try:
            yield
        finally:
            self._call_limit = previous

    def _check_call_limit(self):
        if self._call_limit is not None and self.stats.calls >= self._call_limit:
            raise CallBudgetExceeded('Search request budget exhausted (including retries)')

    @abc.abstractmethod
    def _generate(self, prompt: str) -> str:
        ...

    def generate(self, prompt: str) -> str:
        # Never repeat a successful response just because it was slow. All
        # attempts, including failures, contribute to the request/time counters.
        retries = max(0, int(os.environ.get("LLM_RETRIES", "2")))
        for attempt in range(1, retries + 2):
            self._check_call_limit()
            start = time.perf_counter()
            try:
                response = self._generate(prompt)
            except CallBudgetExceeded:
                raise
            except Exception as e:
                elapsed = time.perf_counter() - start
                self.stats.record(prompt, '', elapsed, error=e)
                # Authentication, invalid parameters and other permanent HTTP
                # errors must not spend the retry allowance.
                status = getattr(e, 'status_code', getattr(e, 'code', None))
                if isinstance(status, int) and status not in (408, 409, 429) and status < 500:
                    raise
                if attempt <= retries:
                    self._check_call_limit()
                    backoff_start = time.perf_counter()
                    time.sleep(0.5 * attempt)
                    self.stats.backoff_time += time.perf_counter() - backoff_start
                    continue
                raise
            elapsed = time.perf_counter() - start
            # Attach model hint to stats for better token counting (best-effort)
            if hasattr(self, "model"):
                model_hint = getattr(self, "model")
                # map known provider model names to a tokenizer encoding when appropriate
                MODEL_TOKENIZER_HINTS = {
                    "gemma": "cl100k_base",
                    "ollama": "cl100k_base",
                    "gpt": None,
                }
                if isinstance(model_hint, str):
                    low = model_hint.lower()
                    for key, hint in MODEL_TOKENIZER_HINTS.items():
                        if key in low:
                            model_hint = hint or model_hint
                            break
            setattr(self.stats, "_model_for_token_count", model_hint if 'model_hint' in locals() else None)
            self.stats.record(prompt, response, elapsed)
            return response


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        super().__init__()
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        self.model = model
        self.temperature = temperature

    def _generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""


class OpenAICompatibleBackend(LLMBackend):
    # works with any provider exposing an OpenAI-style /chat/completions endpoint
    def __init__(self, model: str, base_url: str, api_key_env: str | None = None, temperature: float = 0.0):
        super().__init__()
        import openai
        api_key = os.environ.get(api_key_env, "not-needed") if api_key_env else "not-needed"
        if api_key_env and api_key == "not-needed":
            raise RuntimeError(f"Set the {api_key_env} environment variable first.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model = model
        self.temperature = temperature

    def _generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""


class GroqBackend(OpenAICompatibleBackend):
    # free tier, no credit card. console.groq.com for a key, export GROQ_API_KEY
    def __init__(self, model: str = "openai/gpt-oss-20b", temperature: float = 0.0):
        super().__init__(model=model, base_url="https://api.groq.com/openai/v1",
                          api_key_env="GROQ_API_KEY", temperature=temperature)


class OllamaBackend(OpenAICompatibleBackend):
    # fully local: `ollama pull llama3.1` then `ollama serve`
    def __init__(self, model: str = "llama3.1", temperature: float = 0.0):
        super().__init__(model=model, base_url="http://localhost:11434/v1",
                          api_key_env=None, temperature=temperature)


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-3-5-haiku-20241022", max_tokens: int = 300):
        super().__init__()
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)
        self.model = model
        self.max_tokens = max_tokens

    def _generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


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
    "tweets": 0.60, "fiction": 0.70, "telephone": 0.66, "slate": 0.60,
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
        return label if correct else (("Negative" if label == "Positive" else "Positive") if label in ["Positive", "Negative"] else ("Contradiction" if label == "Entailment" else "Entailment"))

    def _propose_instruction(self, meta_prompt: str) -> str:
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
        return " ".join(ordered)
