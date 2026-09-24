import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: plot_csv.py path/to/reproc.csv')
    raise SystemExit(1)

p = Path(sys.argv[1])
if not p.exists():
    print('File not found:', p)
    raise SystemExit(2)

df = pd.read_csv(p)
# ensure numeric
for col in ['dev_score','source_test_acc','ood_gap','estimated_cost_usd','api_calls']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# plot source_test_acc vs estimated_cost_usd
fig, ax = plt.subplots(figsize=(8,5))
for name, g in df.groupby('name'):
    ax.scatter(g['estimated_cost_usd'], g['source_test_acc'], label=name)
ax.set_xlabel('Estimated cost (USD)')
ax.set_ylabel('Source test accuracy')
ax.legend()
out1 = p.with_name(p.stem + '_src_vs_cost.png')
fig.tight_layout()
fig.savefig(out1)
print('Wrote', out1)

# plot ood_gap vs estimated_cost_usd
fig2, ax2 = plt.subplots(figsize=(8,5))
for name, g in df.groupby('name'):
    ax2.scatter(g['estimated_cost_usd'], g['ood_gap'], label=name)
ax2.set_xlabel('Estimated cost (USD)')
ax2.set_ylabel('OOD gap')
ax2.legend()
out2 = p.with_name(p.stem + '_oodgap_vs_cost.png')
fig2.tight_layout()
fig2.savefig(out2)
print('Wrote', out2)
