"""Fresh, balanced round-sweep splits with exact and shared-span exclusions."""
import random
from dataclasses import asdict
from pathlib import Path

from core import Example, fingerprint, save_json
from data import read_arrow
from fresh_data import ROOT, collect_exclusions, normalized
from improvement_pilot import read
from protocol import validate_splits_manifest

SIZES = dict(train=32, validation=32, confirmation=64, source_test=32,
             amazon_test=64, tweets_test=64)
SPLIT_SEED = 10132026


def spans(text):
    words = normalized(text).split()
    return {tuple(words[i:i+8]) for i in range(len(words)-7)}


def validate(manifest):
    validate_splits_manifest(manifest)
    if manifest['sizes'] != SIZES or manifest['seed'] != SPLIT_SEED:
        raise ValueError('Round sweep split configuration changed')
    if fingerprint(manifest['excluded_texts']) != manifest['exclusion_hash']:
        raise ValueError('Changed exclusion list')
    if set(manifest['splits']) != set(SIZES):
        raise ValueError('Unexpected split names')
    seen = set(manifest['excluded_texts'])
    used_spans = set().union(*(spans(t) for t in seen))
    for name, rows in manifest['splits'].items():
        domain = 'amazon' if name == 'amazon_test' else 'tweets' if name == 'tweets_test' else 'sst2'
        if len(rows) != SIZES[name] or sum(r['label'] == 'Positive' for r in rows) != len(rows)//2:
            raise ValueError('Wrong size or label balance: ' + name)
        for row in rows:
            text = normalized(row['text'])
            if row['domain'] != domain or row['split'] != name:
                raise ValueError('Wrong split metadata')
            if domain == 'sst2' and len(text.split()) < 12:
                raise ValueError('Short SST-2 fragment')
            if text in seen or spans(text) & used_spans:
                raise ValueError('Previously exposed or overlapping text/span')
            seen.add(text)
            used_spans.update(spans(text))


def sample(table, domain, field, labels, name, seed, excluded, used_spans):
    indices = list(range(table.num_rows))
    random.Random(seed).shuffle(indices)
    quotas = dict(Positive=SIZES[name]//2, Negative=SIZES[name]//2)
    rows = []
    for i in indices:
        label = labels.get(table['label'][i].as_py())
        if not quotas.get(label):
            continue
        text = table[field][i].as_py().strip()
        if domain == 'amazon':
            text = text[:500]
        norm = normalized(text)
        if not norm or (domain == 'sst2' and len(norm.split()) < 12):
            continue
        shared = spans(text)
        if norm in excluded or shared & used_spans:
            continue
        excluded.add(norm)
        used_spans.update(shared)
        rows.append(asdict(Example(f'{domain}:rounds:{name}:{i}', text, label, domain, name)))
        quotas[label] -= 1
        if not any(quotas.values()):
            return rows
    raise ValueError('Insufficient fresh examples: ' + name)


def prepare(output):
    output = Path(output)
    path = output/'splits.json'
    if path.exists():
        manifest = read(path)
        validate(manifest)
        return manifest
    paths = [p for p in (ROOT/'output').glob('*/splits.json') if p.resolve() != path.resolve()]
    for folder in ('prompt_opt_harness/logs', 'prompt_opt_harness/prompt_opt_harness/logs',
                   'prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/logs'):
        paths.extend((ROOT/folder).glob('*.json'))
    excluded, provenance = collect_exclusions(paths)
    frozen = sorted(excluded)
    used_spans = set().union(*(spans(t) for t in excluded))
    cache = Path.home()/'.cache/huggingface/datasets'
    def locate(folder, filename, config=None):
        found = [p for p in (cache/folder).rglob(filename) if config is None or config in p.parts]
        if len(found) != 1:
            raise ValueError('Missing or ambiguous cached dataset: ' + folder)
        return found[0]
    sources = dict(sst2=locate('nyu-mll___glue', 'glue-train.arrow', 'sst2'),
                   amazon=locate('fancyzhx___amazon_polarity', 'amazon_polarity-test.arrow'),
                   tweets=locate('cardiffnlp___tweet_eval', 'tweet_eval-test.arrow', 'sentiment'))
    tables = {key:read_arrow(value) for key,value in sources.items()}
    splits = {}
    for i, name in enumerate(SIZES):
        domain = 'amazon' if name == 'amazon_test' else 'tweets' if name == 'tweets_test' else 'sst2'
        splits[name] = sample(tables[domain], domain,
            dict(sst2='sentence', amazon='content', tweets='text')[domain],
            {0:'Negative', 2:'Positive'} if domain == 'tweets' else {0:'Negative', 1:'Positive'},
            name, SPLIT_SEED+i, excluded, used_spans)
    manifest = dict(seed=SPLIT_SEED, sizes=SIZES, splits=splits, split_hash=fingerprint(splits),
        excluded_texts=frozen, exclusion_hash=fingerprint(frozen), exclusion_sources=provenance,
        source_files={k:str(v) for k,v in sources.items()},
        source_sampling='SST-2 official training pool, minimum 12 words; not official validation',
        overlap_rule='case/whitespace-normalized exact text or any shared contiguous 8-word span')
    validate(manifest)
    save_json(path, manifest)
    return manifest
