"""Prompt-trait analysis for research questions Q2/Q3.

Q2: which prompt characteristics predict robustness (small OOD gap, high
worst-case accuracy, low cross-domain variance)?
Q3: which structural or semantic factors (length, specificity, domain
vocabulary) affect OOD transfer?

``prompt_traits`` measures one instruction. ``trait_table`` joins traits with
the OOD outcomes stored in harness run logs, and ``correlations`` reports
Spearman rank correlations between every trait and every outcome.

Standard library only, like the offline demo. Correlations over a handful of
runs are descriptive, not evidence: report ``n`` alongside every value.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
# Phrases that add a decision rule or extra guidance beyond the bare task.
_GUIDANCE_RE = re.compile(
    r"\b(if|unless|when|even if|ignore|consider|focus|pay attention|note that|"
    r"rather than|instead of|do not|don't|avoid|only|must)\b")
_EXAMPLE_RE = re.compile(r"(\be\.?g\.?(?=\s)|\bfor example\b|\bsuch as\b|[\"“][^\"”]{2,40}[\"”])")
STOPWORDS = frozenset("""
a an the and or but if of to in on for with as at by from is are be was were it its this that
these those your you their them they he she we our not no do does did so than then there here
into about over only just whether which what who whom how all any each every one two either
""".split())

TRAIT_NAMES = (
    "words", "sentences", "guidance_clauses", "names_labels", "has_examples",
    "lexical_diversity", "source_vocab_rate", "copied_4grams",
)
OUTCOME_NAMES = ("ood_gap", "relative_ood_gap", "worst_case_acc", "variance")


_ABBREV_RE = re.compile(r"\b(e\.g|i\.e|etc|vs)\.", re.I)


def _normalize(text: str) -> str:
    # "e.g." is one word and does not end a sentence.
    return _ABBREV_RE.sub(lambda m: m.group(1).replace(".", ""), text)


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(_normalize(text).lower())


def _content(words: Iterable[str]) -> List[str]:
    return [w for w in words if w not in STOPWORDS and not w.isdigit()]


def _ngrams(words: Sequence[str], n: int) -> set:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def prompt_traits(instruction: str, labels: Sequence[str] = (),
                  source_texts: Optional[Iterable[str]] = None,
                  target_texts: Optional[Iterable[str]] = None) -> Dict[str, Optional[float]]:
    """Measure one instruction.

    ``source_vocab_rate``: share of the instruction's content words that occur
    in source-domain texts but in no target-domain text. A high value means
    the prompt leans on source-specific vocabulary, the "domain overfitting"
    pattern from the proposal (e.g. product words in a sentiment prompt).
    ``copied_4grams``: 4-word spans copied verbatim from source texts, a
    dev-set memorization signal. Both need example texts and are ``None``
    without them.
    """
    words = _words(instruction)
    content = _content(words)
    traits: Dict[str, Optional[float]] = {
        "words": float(len(words)),
        "sentences": float(len(_SENTENCE_RE.findall(_normalize(instruction).strip() + " "))) if words else 0.0,
        "guidance_clauses": float(len(_GUIDANCE_RE.findall(instruction.lower()))),
        "names_labels": (sum(l.lower() in instruction.lower() for l in labels) / len(labels)) if labels else None,
        "has_examples": float(bool(_EXAMPLE_RE.search(instruction.lower()))),
        "lexical_diversity": (len(set(content)) / len(content)) if content else 0.0,
        "source_vocab_rate": None,
        "copied_4grams": None,
    }
    if source_texts is not None:
        source_texts = list(source_texts)
        source_words = [_words(t) for t in source_texts]
        if target_texts is not None:
            source_vocab = {w for ws in source_words for w in ws}
            target_vocab = {w for t in target_texts for w in _words(t)}
            specific = source_vocab - target_vocab
            traits["source_vocab_rate"] = (sum(w in specific for w in content) / len(content)) if content else 0.0
        source_grams = set().union(*(_ngrams(ws, 4) for ws in source_words)) if source_words else set()
        traits["copied_4grams"] = float(len(_ngrams(words, 4) & source_grams))
    return traits


def load_runs(paths: Iterable[str | Path]) -> List[dict]:
    """Read harness run logs (``logger.save_run`` JSON); directories are globbed."""
    runs = []
    for path in paths:
        path = Path(path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            data = json.loads(file.read_text())
            if isinstance(data, dict) and "best_instruction" in data and data.get("ood_accs"):
                data["_file"] = str(file)
                runs.append(data)
    return runs


def run_outcomes(run: dict) -> Dict[str, float]:
    source = run["source_test_acc"]
    return {
        "ood_gap": run["ood_gap"],
        # Proposal: relative gaps matter when baselines differ (90->80 vs 40->30).
        "relative_ood_gap": run["ood_gap"] / source if source else math.nan,
        "worst_case_acc": run["worst_case_acc"],
        "variance": run["variance"],
    }


def trait_table(runs: Iterable[dict], labels: Sequence[str] = (),
                source_texts: Optional[Iterable[str]] = None,
                target_texts: Optional[Iterable[str]] = None) -> List[dict]:
    source_texts = list(source_texts) if source_texts is not None else None
    target_texts = list(target_texts) if target_texts is not None else None
    rows = []
    for run in runs:
        run_labels = labels or tuple(run.get("task_spec", {}).get("labels", ()))
        row = {"name": run["name"], "instruction": run["best_instruction"]}
        row.update(prompt_traits(run["best_instruction"], run_labels, source_texts, target_texts))
        row.update(run_outcomes(run))
        rows.append(row)
    return rows


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman's rho with average ranks for ties; None if undefined."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else None


def correlations(rows: Sequence[dict]) -> List[dict]:
    """One row per trait: rho with each outcome and the number of runs used."""
    table = []
    for trait in TRAIT_NAMES:
        values = [r.get(trait) for r in rows]
        if all(v is None for v in values):
            continue
        entry = {"trait": trait, "n": sum(v is not None for v in values)}
        for outcome in OUTCOME_NAMES:
            entry[outcome] = spearman(values, [r[outcome] for r in rows])
        table.append(entry)
    return table


def format_markdown(rows: Sequence[dict], corr: Sequence[dict]) -> str:
    def fmt(v):
        return "n/a" if v is None else (f"{v:+.2f}" if isinstance(v, float) else str(v))

    lines = ["## Spearman correlation: prompt trait vs OOD outcome", "",
             "| Trait | n | " + " | ".join(OUTCOME_NAMES) + " |",
             "|---|---|" + "---|" * len(OUTCOME_NAMES)]
    for c in corr:
        lines.append(f"| {c['trait']} | {c['n']} | " + " | ".join(fmt(c[o]) for o in OUTCOME_NAMES) + " |")
    n = len(rows)
    lines += ["", f"_{n} runs. Positive rho with `ood_gap` means the trait goes with a LARGER gap "
              "(worse transfer); positive rho with `worst_case_acc` means better worst-case accuracy._"]
    if n < 10:
        lines.append(f"_Only {n} runs: treat these as descriptive. Run more optimizers x seeds before drawing conclusions._")
    lines += ["", "## Per-run traits", "",
              "| Run | words | guidance | source_vocab_rate | ood_gap | worst_case |", "|---|---|---|---|---|---|"]
    def num(v, signed=False):
        if v is None:
            return "n/a"
        if float(v).is_integer() and not signed:
            return str(int(v))
        return f"{v:+.2f}" if signed else f"{v:.2f}"

    for r in rows:
        lines.append(f"| {r['name']} | {num(r['words'])} | {num(r['guidance_clauses'])} | "
                     f"{num(r['source_vocab_rate'])} | {num(r['ood_gap'], True)} | {num(r['worst_case_acc'])} |")
    return "\n".join(lines)
