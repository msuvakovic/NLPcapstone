"""Reprocess compare JSON results to recompute estimated_cost_usd using current PRICE_* env vars
and optionally re-tokenize per-call traces if present.

Usage:
  python -B tools/reprocess_results.py path/to/compare_*.json

Produces <orig>.reproc.json and <orig>.reproc.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import os


def price_per_1k_for(backend_name: str | None) -> float:
    env_map = {
        "GROQ": float(os.environ.get("PRICE_GROQ", "0.003")),
        "OPENAI": float(os.environ.get("PRICE_OPENAI", "0.03")),
        "OLLAMA": float(os.environ.get("PRICE_OLLAMA", "0.001")),
        "MOCK": float(os.environ.get("PRICE_MOCK", "0.0")),
    }
    if backend_name is None:
        return float(os.environ.get("PRICE_DEFAULT", "0.01"))
    key = backend_name.upper()
    return env_map.get(key, float(os.environ.get("PRICE_DEFAULT", "0.01")))


def reprocess(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    for r in data:
        tokens = int(r.get('input_tokens', 0)) + int(r.get('output_tokens', 0))
        backend_name = r.get('real_backend') or r.get('backend') or r.get('name')
        price = price_per_1k_for(backend_name)
        r['estimated_cost_usd'] = tokens / 1000.0 * price
    out_json = path.with_name(path.stem + '.reproc.json')
    out_csv = path.with_name(path.stem + '.reproc.csv')
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    with open(out_csv, 'w', encoding='utf-8') as fh:
        keys = ['name','optimizer_class','budget','seed','dev_score','source_test_acc','ood_gap','api_calls','input_tokens','output_tokens','estimated_cost_usd']
        fh.write(','.join(keys) + '\n')
        for r in data:
            fh.write(','.join(str(r.get(k,'')) for k in keys) + '\n')
    print('Updated', path)
    print('Wrote CSV to', out_csv)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: reprocess_results.py path/to/compare.json')
        raise SystemExit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print('File not found:', p)
        raise SystemExit(2)
    reprocess(p)
