from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from ..prompts import build_classification_prompt


@dataclass
class Candidate:
    instruction: str
    dev_score: Optional[float] = None
    ood_accs: Optional[Dict[str, float]] = None
    ood_score: Optional[float] = None


def evaluate(backend, instruction: str, examples) -> float:
    if not examples:
        return 0.0
    correct = 0
    # simple dedupe/cache to avoid re-sending identical prompts
    cache = {}
    # Try to enable embedding-based semantic dedupe if sentence-transformers is available.
    embedder = None
    np = None
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        embedder = None
    embeddings = []
    emb_prompts = []
    sim_threshold = 0.92
    for ex in examples:
        prompt = build_classification_prompt(instruction, ex.text)
        if prompt in cache:
            prediction = cache[prompt]
        else:
            # semantic dedupe: check previous prompts for high similarity
            if embedder is not None:
                try:
                    v = embedder.encode(prompt, convert_to_numpy=True)
                    if embeddings:
                        sims = np.dot(embeddings, v) / (np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(v) + 1e-12))
                        best_idx = int(np.argmax(sims))
                        if sims[best_idx] >= sim_threshold:
                            # reuse prediction from most similar prompt
                            prediction = cache[emb_prompts[best_idx]]
                            cache[prompt] = prediction
                            if prediction.startswith(ex.label.lower()[:3]):
                                correct += 1
                            continue
                except Exception:
                    # fall back to exact dedupe
                    pass
            prediction = backend.generate(prompt).strip().lower()
            cache[prompt] = prediction
            if embedder is not None:
                try:
                    embeddings.append(v)
                    emb_prompts.append(prompt)
                except Exception:
                    pass
        if prediction.startswith(ex.label.lower()[:3]):
            correct += 1
    return correct / len(examples)


class Optimizer:
    name = "base"

    def __init__(self, backend, budget: int, seed: int = 0):
        self.backend = backend
        self.budget = budget  # max LLM calls for this run
        self.seed = seed
        self.history: List[Candidate] = []

    def ood_rank_score(self, dev_score: float, ood_accs: dict | list | tuple) -> float:
        if isinstance(ood_accs, dict):
            ood_values = list(ood_accs.values())
        else:
            ood_values = list(ood_accs)
        if not ood_values:
            return dev_score
        mean_ood = sum(ood_values) / len(ood_values)
        worst_ood = min(ood_values)
        gap = abs(dev_score - mean_ood)
        # score = 0.4 * source_acc + 0.6 * mean_ood_acc - ? * |ood_gap|
        return 0.4 * dev_score + 0.6 * mean_ood - 0.5 * gap

    def progressive_score(self, dev_score: float, ood_accs: dict | list | tuple, progress: float) -> float:
        ood_score = self.ood_rank_score(dev_score, ood_accs)
        return (1.0 - progress) * dev_score + progress * ood_score

    def _dev_score(self, instruction: str, dev_examples) -> float:
        return evaluate(self.backend, instruction, dev_examples)

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        raise NotImplementedError

def evaluate_per_example(backend, instruction: str, examples) -> List[Tuple[object, str, bool]]:
    if not examples:
        return []
    results = []
    cache = {}
    # same embedding dedupe strategy as evaluate()
    embedder = None
    np = None
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        embedder = None
    embeddings = []
    emb_prompts = []
    sim_threshold = 0.92
    for ex in examples:
        prompt = build_classification_prompt(instruction, ex.text)
        if prompt in cache:
            prediction = cache[prompt]
        else:
            if embedder is not None:
                try:
                    v = embedder.encode(prompt, convert_to_numpy=True)
                    if embeddings:
                        sims = np.dot(embeddings, v) / (np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(v) + 1e-12))
                        best_idx = int(np.argmax(sims))
                        if sims[best_idx] >= sim_threshold:
                            prediction = cache[emb_prompts[best_idx]]
                            cache[prompt] = prediction
                            is_correct = prediction.startswith(ex.label.lower()[:3])
                            results.append((ex, prediction, is_correct))
                            continue
                except Exception:
                    pass
            prediction = backend.generate(prompt).strip().lower()
            cache[prompt] = prediction
            if embedder is not None:
                try:
                    embeddings.append(v)
                    emb_prompts.append(prompt)
                except Exception:
                    pass
        is_correct = prediction.startswith(ex.label.lower()[:3])
        results.append((ex, prediction, is_correct))
    return results
