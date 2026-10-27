#!/usr/bin/env bash
# One command for the real experiment: quick check -> full run -> analysis -> demo page.
#
#   bash run_all.sh                               # Qwen2.5-7B on Ollama, 3 seeds, EVERY optimizer that is installed
#   OPTIMIZERS=core bash run_all.sh               # only the six core optimizers (much faster)
#   MAX_TOKENS=1024 LOG_DIR=logs/after_1024 bash run_all.sh  # same run with a bigger answer cap (token-cap check)
#   SKIP_QUICK=1 bash run_all.sh                  # skip the 10-question smoke test
#
# Safe to stop (Ctrl-C) and start again: finished runs are skipped (--resume).
# Settings you can override with environment variables are listed below.
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${BACKEND:-ollama}"
MODEL="${MODEL:-qwen2.5:7b}"
SEEDS="${SEEDS:-3}"
N_PER_SPLIT="${N_PER_SPLIT:-50}"      # questions per split per seed; more = tighter confidence intervals
BUDGET="${BUDGET:-400}"                # API calls each optimizer may spend searching. Scoring one prompt costs N_PER_SPLIT calls,
                                       # so 400 with 50 questions = about 8 scored prompts. Keep BUDGET at 6x N_PER_SPLIT or more.
OOD="${OOD:-svamp,gsmplus}"            # new datasets. Two or more are needed for cross-domain variance. OOD=svamp uses the original one only.
MAX_TOKENS="${MAX_TOKENS:-512}"
PROPOSAL_TEMP="${PROPOSAL_TEMP:-0.8}"
OPTIMIZERS="${OPTIMIZERS:-all}"        # all | core | extended | dspy | names such as opro,gepa,dspy-gepa. Missing packages are skipped with a note.
LOG_DIR="${LOG_DIR:-logs/after}"
OUT_DIR="${OUT_DIR:-analysis/$(basename "$LOG_DIR")}"
SKIP_QUICK="${SKIP_QUICK:-0}"

COMMON="--backend $BACKEND --model $MODEL --start bare --proposal-temp $PROPOSAL_TEMP --optimizers $OPTIMIZERS --max-tokens $MAX_TOKENS --ood $OOD"

if [ "$BUDGET" -lt $(( 6 * N_PER_SPLIT )) ]; then
  echo "!! Warning: BUDGET=$BUDGET is below 6 x N_PER_SPLIT=$N_PER_SPLIT, so each optimizer can score fewer than 6 prompts. Raise BUDGET or lower N_PER_SPLIT."
fi
python3 - "$OPTIMIZERS" <<'PY'
import sys
from prompt_opt_harness.registry import available, resolve
specs = resolve(sys.argv[1])
ok = [s.name for s in specs if available(s)[0]]
missing = [(s.name, available(s)[1]) for s in specs if not available(s)[0]]
print(f"== Optimizers that will run ({len(ok)}): " + ", ".join(ok))
for name, why in missing:
    print(f"   skipped {name}: {why}")
if missing:
    print("   To include them, install what is listed above and run again (finished runs are kept).")
PY
echo "== Settings: $BACKEND / $MODEL | $SEEDS seeds | $N_PER_SPLIT questions per split | budget $BUDGET | max_tokens $MAX_TOKENS | new data: $OOD | logs in $LOG_DIR"

# 0. Is the model reachable?
if [ "$BACKEND" = "ollama" ]; then
  if ! curl -s --max-time 5 http://localhost:11434/api/tags > /tmp/ollama_tags.json 2>/dev/null; then
    echo "!! Ollama is not running. Open the Ollama app (or run 'ollama serve' in another Terminal window) and try again."
    exit 1
  fi
  if ! grep -q "\"${MODEL}\"" /tmp/ollama_tags.json; then
    echo "!! Model $MODEL is not downloaded. Run: ollama pull $MODEL"
    exit 1
  fi
fi
python3 -c "import datasets, openai" 2>/dev/null || { echo "!! Missing packages. Run: pip3 install datasets openai"; exit 1; }

# 1. Quick check: tiny run to prove everything works end to end before spending hours.
if [ "$SKIP_QUICK" != "1" ]; then
  echo "== 1/4 Quick check (1 seed, 10 questions, budget 60)"
  rm -rf logs/quick_check
  START=$(date +%s)
  python3 run_groq_reasoning_real_data.py $COMMON --seeds 1 --n-per-split 10 --budget 60 --log-dir logs/quick_check
  ELAPSED=$(( $(date +%s) - START ))
  python3 - "$ELAPSED" "$SEEDS" "$N_PER_SPLIT" "$BUDGET" "$OPTIMIZERS" "$OOD" <<'PY'
import glob, json, sys
from prompt_opt_harness.registry import available, resolve
elapsed, seeds, n, budget = (int(x) for x in sys.argv[1:5])
specs = [s for s in resolve(sys.argv[5]) if available(s)[0]]
n_ood = len(sys.argv[6].split(","))
calls = sum(json.load(open(f))["api_calls"] for f in glob.glob("logs/quick_check/*.json"))
sec = elapsed / max(calls, 1)
eval_calls = n * (1 + n_ood)                      # source test + each new dataset
per_seed = 2 * (n + eval_calls) + sum(s.budget_mult * budget + eval_calls for s in specs)
guarded = sum(1 for s in specs if s.cls.needs_confirm)
full = seeds * (per_seed + guarded * 2 * n)       # guard: starting prompt and winner on n confirmation questions
print(f"== Quick check: {calls} calls in {elapsed}s ({sec:.1f}s per call, includes loading the datasets, so the estimate below is on the high side).")
print(f"== Full run: {len(specs)} optimizers x {seeds} seeds, about {full} calls, roughly {full * sec / 3600:.1f} hours on this machine.")
print("   Too long? Use OPTIMIZERS=core, or lower SEEDS, N_PER_SPLIT or BUDGET (keep BUDGET >= 6 x N_PER_SPLIT), e.g.  OPTIMIZERS=core SEEDS=2 N_PER_SPLIT=40 BUDGET=300 bash run_all.sh")
PY
  python3 analyze.py --logs logs/quick_check --out analysis/quick_check > /dev/null
  echo "== Quick-check truncation: $(grep -o 'Verdict: [A-Z]*' analysis/quick_check/results.md | head -1)  (see analysis/quick_check/results.md)"
fi

# 2. Full run (resumable).
echo "== 2/4 Full run"
python3 run_groq_reasoning_real_data.py $COMMON --seeds "$SEEDS" --n-per-split "$N_PER_SPLIT" --budget "$BUDGET" --log-dir "$LOG_DIR" --resume

# 3. Analysis: metrics, confidence intervals, truncation audit, findings.
echo "== 3/4 Analysis"
python3 analyze.py --logs "$LOG_DIR" --out "$OUT_DIR"

# 4. Demo page for the career fair.
echo "== 4/4 Demo page"
python3 make_demo.py "$LOG_DIR" --out "$OUT_DIR/demo.html"   # shows 5 optimizers; pass --methods to choose which (see make_demo.py --help)

echo
echo "Done. Open these:"
echo "  $OUT_DIR/results.md     numbers and auto-written findings (paste into the report)"
echo "  $OUT_DIR/fig_gap.svg    the overfitting figure"
echo "  $OUT_DIR/demo.html      the walkthrough demo built from these real runs"
