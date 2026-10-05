"""Frozen, balanced splits from the already downloaded public HF datasets."""
import json
import random
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa

from core import Example, fingerprint, save_json
from protocol import validate_splits_manifest


def read_arrow(path):
    with pa.memory_map(str(path), 'r') as f:
        return pa.ipc.open_stream(f).read_all()


def choose(table, domain, field, label_map, n, seed, split, excluded, max_chars=None):
    indices = list(range(table.num_rows))
    random.Random(seed).shuffle(indices)
    quotas = {'Positive': n//2, 'Negative': n//2}
    result = []
    for i in indices:
        label = label_map.get(table['label'][i].as_py())
        if label not in quotas or quotas[label] == 0:
            continue
        text = table[field][i].as_py().strip()
        if max_chars:
            text = text[:max_chars]
        normalized = ' '.join(text.lower().split())
        if normalized in excluded:
            continue
        excluded.add(normalized)
        result.append(Example(f'{domain}:{split}:{i}', text, label, domain, split))
        quotas[label] -= 1
        if not any(quotas.values()):
            break
    if len(result) != n:
        raise ValueError(f'Insufficient unique labeled examples for {domain}/{split}')
    random.Random(seed+1).shuffle(result)
    return result


def prepare(output, train_n=64, val_n=64, test_n=300, split_seed=2712026):
    target = Path(output)/'splits.json'
    if target.exists():
        obj = json.loads(target.read_text(encoding='utf-8'))
        validate_splits_manifest(obj)
        if obj['config'] != dict(train_n=train_n, val_n=val_n, test_n=test_n, split_seed=split_seed):
            raise ValueError('Saved splits differ from requested configuration')
        return {k:[Example(**e) for e in v] for k,v in obj['splits'].items()}
    root = Path.home()/'.cache/huggingface/datasets'
    def locate(folder, filename, config=None):
        paths = [p for p in (root/folder).rglob(filename) if config is None or config in p.parts]
        if len(paths) != 1:
            raise ValueError(f'Expected one cached source: {folder}/{filename}; got {len(paths)}')
        return paths[0]
    sources = {
        'sst':locate('nyu-mll___glue','glue-validation.arrow','sst2'),
        'amazon':locate('fancyzhx___amazon_polarity','amazon_polarity-test.arrow'),
        'tweets':locate('cardiffnlp___tweet_eval','tweet_eval-test.arrow','sentiment'),
    }
    # One label-balanced, random partition of SST validation avoids the fragment/
    # full-sentence shift induced by treating SST training phrases as test sentences.
    # These are custom source splits, not the official hidden-label SST test set.
    excluded = set()
    splits = {}
    sst = read_arrow(sources['sst'])
    for j,(name,n) in enumerate([('train',train_n),('val',val_n),('test',test_n)]):
        splits['sst_'+name] = choose(sst,'sst2','sentence',{0:'Negative',1:'Positive'},
                                      n,split_seed+j,name,excluded)
    amazon = read_arrow(sources['amazon'])
    # Reserve source splits only for the separately labeled multi-source regime.
    for j,(name,n) in enumerate([('train',train_n),('val',val_n),('test',test_n)]):
        splits['amazon_'+name] = choose(amazon,'amazon','content',{0:'Negative',1:'Positive'},
                                      n,split_seed+10+j,name,excluded,max_chars=500)
    tweets = read_arrow(sources['tweets'])
    splits['tweets_test'] = choose(tweets,'tweets','text',{0:'Negative',2:'Positive'},
                                  test_n,split_seed+20,'test',excluded)
    raw = {k:[asdict(e) for e in v] for k,v in splits.items()}
    save_json(target,dict(config=dict(train_n=train_n,val_n=val_n,test_n=test_n,split_seed=split_seed),
                         source_files={k:str(v) for k,v in sources.items()},
                         split_hash=fingerprint(raw),splits=raw))
    return splits


def balanced_half(examples):
    return [e for label in ('Positive','Negative')
            for e in [x for x in examples if x.label == label][:len(examples)//4]]
