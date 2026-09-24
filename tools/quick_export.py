import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: quick_export.py path/to/compare.json')
    sys.exit(1)

p = Path(sys.argv[1])
if not p.exists():
    print('File not found:', p)
    sys.exit(2)

data = json.loads(p.read_text(encoding='utf-8'))
keys = ['name','optimizer_class','budget','seed','dev_score','source_test_acc','ood_gap','api_calls','input_tokens','output_tokens','estimated_cost_usd']
out = p.with_suffix('.csv')
with open(out, 'w', encoding='utf-8') as fh:
    fh.write(','.join(keys) + '\n')
    for r in data:
        row = [str(r.get(k,'')) for k in keys]
        fh.write(','.join(row) + '\n')
print('Wrote', out)
