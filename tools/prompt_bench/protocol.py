"""Fail closed when a resumed experiment differs from its frozen configuration."""
import json
import platform
from pathlib import Path

from core import fingerprint

ROOT = Path(__file__).resolve().parents[2]
VERSION = 2


def source_hashes():
    paths = [ROOT/'tools/prompt_bench'/name for name in
             ('core.py', 'data.py', 'methods.py', 'run.py', 'protocol.py', 'latency.py')]
    package = ROOT/'prompt_opt_harness/prompt_opt_harness/prompt_opt_harness'
    paths += sorted(package.glob('*.py')) + sorted((package/'optimizers').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): fingerprint(p.read_text(encoding='utf-8')) for p in paths}


def validate_splits_manifest(manifest):
    if fingerprint(manifest['splits']) != manifest.get('split_hash'):
        raise ValueError('Saved split fingerprint mismatch; do not resume changed data')
    ids, texts = set(), set()
    for rows in manifest['splits'].values():
        for row in rows:
            text = ' '.join(row['text'].casefold().split())
            if row['id'] in ids or text in texts:
                raise ValueError('Saved splits contain overlapping IDs or normalized texts')
            if row['label'] not in ('Positive', 'Negative'):
                raise ValueError('Saved split label outside the sentiment task')
            ids.add(row['id'])
            texts.add(text)
    return manifest['split_hash']


def build_protocol(args, split_manifest):
    split_hash = validate_splits_manifest(split_manifest)
    return dict(protocol_version=VERSION, model=args.model,
                endpoint=args.endpoint.rstrip('/'), python=platform.python_version(), gepa='0.1.4',
                seeds=args.seeds, regimes=args.regimes, methods=args.methods,
                evaluation_cap=args.budget, proposal_cap=16, workers=args.workers,
                max_actual_calls_per_session=20000,
                source_train_n=args.train_n, source_val_n=args.val_n, test_per_domain=args.test_n,
                split_config=split_manifest['config'], split_hash=split_hash,
                task=dict(name='sentiment', labels=['Positive', 'Negative']),
                decoding=dict(temperature=0.0, seed=12345, max_tokens=16),
                proposal_decoding=dict(temperature=0.7, max_tokens=768, reflection_max_tokens=2048),
                selection_rules=['mean', 'stable', 'worst_group (multi only)'],
                phase='search_before_final_test', source_file_hashes=source_hashes())


def validate_artifact_bindings(output, protocol):
    """Check provenance even for files a resumed run would otherwise skip."""
    output = Path(output)
    protocol_hash = fingerprint(protocol)
    paths = list(output.glob('*/search.json')) + list((output/'final_predictions').glob('*.json'))
    for path in paths:
        artifact = json.loads(path.read_text(encoding='utf-8'))
        if artifact.get('protocol_hash') != protocol_hash:
            raise ValueError(f'Artifact does not match the frozen protocol: {path.name}')
    return protocol_hash


def lock_protocol(output, config, stage):
    output = Path(output)
    manifest = output/'protocol.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved.get('protocol_version') != VERSION:
            raise ValueError('Legacy experiment lacks a versioned protocol lock. '
                             'Keep it read-only and use a new output directory for model runs.')
        differences = sorted(key for key in saved.keys() | config.keys() if saved.get(key) != config.get(key))
        if differences:
            raise ValueError('Experiment configuration changed: ' + ', '.join(differences) +
                             '. Use a new output directory; existing results were not modified.')
    else:
        occupied = (list(output.glob('*/search.json')) + list(output.glob('requests.sqlite*')) +
                    list((output/'final_predictions').glob('*.json')) +
                    [p for p in (output/'frozen_prompts.json', output/'latency.json') if p.exists()])
        if occupied or stage not in ('prepare', 'search'):
            raise ValueError('Missing protocol for existing or evaluation-only experiment; use a new output directory')
        # Exclusive creation prevents two initializers from replacing the lock.
        try:
            with manifest.open('x', encoding='utf-8') as stream:
                json.dump(config, stream, ensure_ascii=False, indent=2)
        except FileExistsError:
            return lock_protocol(output, config, stage)
    return validate_artifact_bindings(output, config)


def validate_execution_context(output):
    """Validate the stored configuration against current code/data for side runners."""
    output = Path(output)
    saved = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    if saved.get('protocol_version') != VERSION:
        raise ValueError('Legacy experiment is read-only; use a new output directory for model runs')
    split = json.loads((output/'splits.json').read_text(encoding='utf-8'))
    expected = dict(saved, split_hash=validate_splits_manifest(split), split_config=split['config'],
                    source_file_hashes=source_hashes(), python=platform.python_version())
    lock_protocol(output, expected, 'evaluate')
    return saved
