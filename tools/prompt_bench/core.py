"""Independent, exact-cache benchmark transport and guarded evaluation."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

BASE = 'Classify the sentiment of the following text as Positive or Negative.'
HUMAN = ('Read the text and decide whether the overall sentiment is Positive or Negative. '
         'Consider the overall tone of the text, not just individual words.')


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    domain: str
    split: str


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def task_prompt(instruction, text):
    return f'{instruction.strip()}\n\nText: {text}\nAnswer with exactly one word: Positive or Negative.'


def parse_label(raw):
    # A malformed answer is an error, never a substring/prefix match.
    value = raw.strip().strip('"\'').strip()
    value = re.sub(r'[.!]$', '', value).lower()
    return {'positive': 'Positive', 'negative': 'Negative'}.get(value, 'INVALID')


class BudgetExceeded(RuntimeError):
    pass


class Client:
    def __init__(self, cache_path, model='gemma4:31b-cloud', workers=4,
                 endpoint='http://localhost:11434/v1', max_actual_calls=20000):
        self.model, self.workers, self.endpoint = model, workers, endpoint.rstrip('/')
        self.max_actual_calls = max_actual_calls
        self.lock = threading.RLock()
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.commit()
        self.stats = dict(actual_calls=0, failed_attempts=0, cache_hits=0, input_tokens=0,
                          output_tokens=0, request_seconds=0.0)

    def snapshot(self):
        with self.lock:
            return dict(self.stats)

    def complete(self, prompt, *, temperature=0.0, seed=12345, max_tokens=16, cache=True):
        messages = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]
        payload = dict(model=self.model, messages=messages, temperature=temperature,
                       seed=seed, max_tokens=max_tokens, stream=False)
        key = fingerprint({'endpoint': self.endpoint, 'payload': payload})
        with self.lock:
            row = self.db.execute('SELECT value FROM responses WHERE key=?', (key,)).fetchone() if cache else None
            if row:
                self.stats['cache_hits'] += 1
                return json.loads(row[0])
        last_error = None
        for attempt in range(6):
            with self.lock:
                if self.stats['actual_calls'] >= self.max_actual_calls:
                    raise BudgetExceeded('Global physical-request safety cap reached')
                self.stats['actual_calls'] += 1
            start = time.perf_counter()
            req = urllib.request.Request(self.endpoint + '/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    data = json.load(response)
                content = data['choices'][0]['message'].get('content') or ''
                usage = data.get('usage', {})
                record = dict(text=content, usage=usage, model=data.get('model'),
                              seconds=time.perf_counter()-start,
                              finish_reason=data['choices'][0].get('finish_reason'))
                with self.lock:
                    self.stats['input_tokens'] += usage.get('prompt_tokens', 0)
                    self.stats['output_tokens'] += usage.get('completion_tokens', 0)
                    self.stats['request_seconds'] += record['seconds']
                    if cache:
                        self.db.execute('INSERT OR REPLACE INTO responses VALUES (?,?)',
                                        (key, json.dumps(record)))
                        self.db.commit()
                return record
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                with self.lock:
                    self.stats['failed_attempts'] += 1
                    self.stats['request_seconds'] += time.perf_counter()-start
                print(f'Model request attempt {attempt+1}/6 failed: {type(exc).__name__}; '
                      f'HTTP status={getattr(exc,"code",None)}',flush=True)
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt < 5:
                    time.sleep(min(30, 2**(attempt+1)))
        raise RuntimeError(f'Model request failed after six attempts: {last_error}')


class Evaluator:
    """Optimization only accepts examples from an explicit source-data allowlist.

    Budget counts distinct candidate/example evaluations within this run even if
    cross-run transport caching saves a physical request. Thus cached runs do not
    get a larger search budget. Whole batches are reserved before dispatch.
    """
    def __init__(self, client, allowed, budget=512, proposal_limit=16, seed=0):
        self.client, self.seed = client, seed
        self.allowed = {e.id: e for e in allowed}
        self.budget, self.proposal_limit = budget, proposal_limit
        self.rows = {}
        self.eval_count = 0
        self.proposal_count = 0
        self.proposal_tokens = 0
        self.logical_input_tokens = 0
        self.logical_output_tokens = 0
        self.audit = []
        self.lock = threading.RLock()

    @property
    def remaining(self):
        return self.budget - self.eval_count

    def evaluate(self, instruction, examples):
        with self.lock:
            for e in examples:
                if self.allowed.get(e.id) != e:
                    raise ValueError(f'Example outside optimization allowlist: {e.id}')
            missing = list({(instruction, e.id): e for e in examples
                            if (instruction, e.id) not in self.rows}.values())
            if len(missing) > self.remaining:
                raise BudgetExceeded(f'Batch of {len(missing)} exceeds remaining budget {self.remaining}')
            self.eval_count += len(missing)
        def classify(e):
            record = self.client.complete(task_prompt(instruction, e.text))
            pred = parse_label(record['text'])
            return e, dict(prediction=pred, raw=record['text'], correct=int(pred == e.label),
                          usage=record['usage'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.client.workers) as pool:
            results = list(pool.map(classify, missing))
        with self.lock:
            for e, row in results:
                self.rows[(instruction, e.id)] = row
                self.logical_input_tokens += row['usage'].get('prompt_tokens', 0)
                self.logical_output_tokens += row['usage'].get('completion_tokens', 0)
            self.audit.append(dict(prompt_hash=fingerprint(instruction), example_ids=[e.id for e in examples],
                                   new_evaluations=len(missing)))
            if len(missing) >= 32:
                stats=self.client.snapshot() if hasattr(self.client,'snapshot') else {}
                print(f'  Evaluated {len(missing)} examples; evaluation usage {self.eval_count}/{self.budget}; '
                      f'session requests={stats.get("actual_calls",0)}, failed attempts={stats.get("failed_attempts",0)}', flush=True)
            return [self.rows[(instruction, e.id)] for e in examples]

    def propose(self, prompt, max_tokens=768):
        if self.proposal_count >= self.proposal_limit:
            raise BudgetExceeded('Proposal-call limit reached')
        self.proposal_count += 1
        rec = self.client.complete(prompt, temperature=0.7, seed=self.seed*1000+self.proposal_count,
                                   max_tokens=max_tokens)
        self.proposal_tokens += rec['usage'].get('completion_tokens', 0)
        self.logical_input_tokens += rec['usage'].get('prompt_tokens', 0)
        self.logical_output_tokens += rec['usage'].get('completion_tokens', 0)
        if rec.get('finish_reason') == 'length':
            raise RuntimeError('Truncated proposal; do not silently evaluate an incomplete instruction')
        return rec['text']

    def summary(self):
        return dict(unique_example_evaluations=self.eval_count, evaluation_cap=self.budget,
                    proposal_calls=self.proposal_count, proposal_cap=self.proposal_limit,
                    logical_input_tokens=self.logical_input_tokens,
                    logical_output_tokens=self.logical_output_tokens)


def strip_fence(text):
    blocks = re.findall(r'```(?:[\w-]*\n)?(.*?)```', text, flags=re.S)
    return (blocks[-1] if blocks else text).strip().strip('"')


def difference(before, after):
    return {k: after[k] - before[k] for k in before}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
