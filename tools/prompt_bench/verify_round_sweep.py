"""Audit the frozen round pilot with serialized SQLite replay reads.

The October 13 execution source is hash-locked. Its embedded ReplayClient used
one SQLite connection across workers; concurrent reads returned wrong/empty
rows on this Python 3.13 runtime. Serialize replay only while
preserving the exact executing source and protocol of the completed experiment.
Live transport already protects cache reads with its lock in core.Client.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import round_sweep as sweep
from round_data import validate


class SerialReplayClient(sweep.ReplayClient):
    workers = 1


def verify(output):
    output = Path(output)
    manifest = sweep.read(output/'splits.json')
    validate(manifest)
    config = sweep.read(output/'protocol.json')
    args = SimpleNamespace(model=config['model'], endpoint=config['endpoint'])
    pid = sweep.lock(output, sweep.configuration(args, manifest))
    # This standalone verifier has no inference fallback. The temporary class
    # replacement affects only offline replay concurrency, not payloads/scores.
    with patch.object(sweep, 'ReplayClient', SerialReplayClient):
        result = sweep.audit(output, manifest, pid, config)
    result['replay_workers'] = 1
    result['verifier_hash'] = sweep.fingerprint(Path(__file__).read_text(encoding='utf-8'))
    sweep.save_json(output/'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=sweep.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(sweep.json.dumps(verify(args.output), indent=2))
