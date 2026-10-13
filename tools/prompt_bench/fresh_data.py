"""Fresh pilot splits exclude all examples recoverable from earlier local logs."""
import json
import re
from dataclasses import asdict
from pathlib import Path

from core import Example, fingerprint, save_json
from data import choose, read_arrow
from protocol import validate_splits_manifest

ROOT = Path(__file__).resolve().parents[2]
SIZES = dict(train=64, validation=64, confirmation=80, source_test=64, target_test=160)


def normalized(text):
    return ' '.join(text.casefold().split())


def collect_exclusions(paths):
    texts, provenance = set(), {}
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'text' and isinstance(value, str):
                    texts.add(normalized(value))
                if key == 'prompt' and isinstance(value, str):
                    for match in re.finditer(r'\nText: (.*?)\nAnswer with exactly', value, re.S):
                        texts.add(normalized(match.group(1)))
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    for path in sorted(paths):
        content = path.read_text(encoding='utf-8')
        provenance[str(path.resolve())] = fingerprint(content)
        walk(json.loads(content))
    return texts, provenance


def validate_fresh(manifest):
    validate_splits_manifest(manifest)
    if manifest.get('sizes') != SIZES:
        raise ValueError('Fresh split sizes differ from the pilot schedule')
    if fingerprint(manifest['excluded_texts']) != manifest['exclusion_hash']:
        raise ValueError('Exclusion fingerprint mismatch')
    excluded = set(manifest['excluded_texts'])
    expected = dict(train=64, validation=64, confirmation=80, source_test=64,
                    amazon_test=160, tweets_test=160)
    if set(manifest['splits']) != set(expected):
        raise ValueError('Unexpected fresh split names')
    for name, rows in manifest['splits'].items():
        if len(rows) != expected[name]:
            raise ValueError('Incorrect split size: ' + name)
        if sum(r['label'] == 'Positive' for r in rows) != len(rows)//2:
            raise ValueError('Unbalanced fresh split: ' + name)
        domain = 'amazon' if name == 'amazon_test' else 'tweets' if name == 'tweets_test' else 'sst2'
        if any(r['domain'] != domain or r['split'] != name for r in rows):
            raise ValueError('Incorrect split metadata: ' + name)
        if any(normalized(r['text']) in excluded for r in rows):
            raise ValueError('Previously exposed example in fresh split')


def prepare_fresh(output, seed=1042026):
    output = Path(output)
    target = output/'splits.json'
    if target.exists():
        manifest = json.loads(target.read_text(encoding='utf-8'))
        validate_fresh(manifest)
        if manifest['seed'] != seed:
            raise ValueError('Fresh split seed changed')
        return manifest
    paths = [p for p in (ROOT/'output').glob('*/splits.json') if p.resolve() != target.resolve()]
    # New runs also contain split manifests, so future pilots exclude this pilot.
    for folder in ('prompt_opt_harness/logs', 'prompt_opt_harness/prompt_opt_harness/logs',
                   'prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/logs'):
        paths += list((ROOT/folder).glob('*.json'))
    excluded, provenance = collect_exclusions(paths)
    frozen_exclusions = sorted(excluded)
    cache = Path.home()/'.cache/huggingface/datasets'
    def locate(folder, filename, config=None):
        matches = [p for p in (cache/folder).rglob(filename) if config is None or config in p.parts]
        if len(matches) != 1:
            raise ValueError('Ambiguous or missing cached dataset: ' + folder)
        return matches[0]
    sources = dict(sst=locate('nyu-mll___glue', 'glue-validation.arrow', 'sst2'),
                   amazon=locate('fancyzhx___amazon_polarity', 'amazon_polarity-test.arrow'),
                   tweets=locate('cardiffnlp___tweet_eval', 'tweet_eval-test.arrow', 'sentiment'))
    splits = {}
    table = read_arrow(sources['sst'])
    for i, name in enumerate(('train', 'validation', 'confirmation', 'source_test')):
        splits[name] = choose(table, 'sst2', 'sentence', {0:'Negative', 1:'Positive'},
                              SIZES[name], seed+i, name, excluded)
    for i, domain in enumerate(('amazon', 'tweets')):
        splits[domain+'_test'] = choose(read_arrow(sources[domain]), domain,
            'content' if domain == 'amazon' else 'text',
            {0:'Negative', 1:'Positive'} if domain == 'amazon' else {0:'Negative', 2:'Positive'},
            SIZES['target_test'], seed+10+i, domain+'_test', excluded,
            max_chars=500 if domain == 'amazon' else None)
    raw = {k:[asdict(e) for e in rows] for k, rows in splits.items()}
    manifest = dict(seed=seed, sizes=SIZES, splits=raw, split_hash=fingerprint(raw),
                    excluded_texts=frozen_exclusions, exclusion_hash=fingerprint(frozen_exclusions),
                    exclusion_sources=provenance, source_files={k:str(v) for k,v in sources.items()})
    validate_fresh(manifest)
    save_json(target, manifest)
    return manifest


def examples(manifest, name):
    return [Example(**row) for row in manifest['splits'][name]]
