from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SET_KEYS = ["model", "task", "dataset", "split_mode", "start", "n_dev", "proposal_temperature"]
LABEL_KEYS = ["model", "split_mode", "start", "proposal_temperature", "budget", "n_dev"]
SIMULATED_MARKERS = ("mock", "scripted", "fixture")
RESPONSE_LIMIT = 1600


def describe(key: str, value) -> str:
    if key == "split_mode":
        return {"same": "tuning and test questions from the same split",
                "train_test": "tuning from train split, test from test split"}.get(value, str(value))
    if key == "start":
        return f"{value} starting prompt"
    if key == "proposal_temperature":
        return f"proposal temperature {value}"
    if key == "budget":
        return f"budget {value} calls"
    if key == "n_dev":
        return f"{value} questions per set"
    return str(value)


def load_runs(dirs):
    runs, skipped = [], 0
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                with open(path) as f:
                    run = json.load(f)
            except (OSError, ValueError):
                continue
            if run.get("schema_version", 1) < 2 or "eval_records" not in run:
                skipped += 1
                continue
            for records in list(run["eval_records"].values()) + [run.get("dev_records", [])]:
                for rec in records:
                    if isinstance(rec.get("response"), str) and len(rec["response"]) > RESPONSE_LIMIT:
                        rec["response"] = rec["response"][:RESPONSE_LIMIT] + " ..."
            run["_path"] = path
            runs.append(run)
    return runs, skipped


def group_sets(runs):
    sets = {}
    for run in runs:
        cfg = run.get("config", {})
        key = tuple(json.dumps(cfg.get(k)) for k in SET_KEYS)
        sets.setdefault(key, []).append(run)

    out = []
    for key, members in sets.items():
        latest = {}
        for run in members:
            ident = (run["name"], run.get("seed", 0))
            if ident not in latest or run["timestamp"] > latest[ident]["timestamp"]:
                latest[ident] = run
        kept = sorted(latest.values(), key=lambda r: (r.get("seed", 0), r["name"]))
        cfg = dict(kept[0].get("config", {}))
        cfg["budget"] = max(r.get("config", {}).get("budget") or 0 for r in kept)
        backend = str(cfg.get("backend", "")).lower() + " " + str(cfg.get("model", "")).lower()
        out.append({
            "config": cfg,
            "runs": kept,
            "newest": max(r["timestamp"] for r in kept),
            "simulated": any(m in backend for m in SIMULATED_MARKERS),
        })

    out.sort(key=lambda s: s["newest"], reverse=True)
    for i, s in enumerate(out):
        differing = [k for k in LABEL_KEYS if len({json.dumps(o["config"].get(k)) for o in out}) > 1]
        s["label"] = ", ".join(describe(k, s["config"].get(k)) for k in differing) or str(s["config"].get("model", "run set"))
        s["id"] = i
    return out


DEFAULT_SHOWN = ["OPRO", "EvoPrompt-lite", "GEPA-lite", "TextGrad-lite", "Random-search"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained demo page from run logs.")
    parser.add_argument("log_dirs", nargs="*", default=["logs"])
    parser.add_argument("--out", default="demo.html")
    parser.add_argument("--title", default=None)
    parser.add_argument("--headline", default=None, help="one sentence shown under the summary table")
    parser.add_argument("--footer", default=None)
    parser.add_argument("--methods", default=None,
                        help="comma-separated optimizers to show (at most 5; the page has 5 colors). Default: OPRO, EvoPrompt-lite, GEPA-lite, "
                             "TextGrad-lite, Random-search. Zero-shot and Human-written are always shown. Everything stays in analyze.py's report.")
    parser.add_argument("--next-step", action="append", default=None, help="a line for the Next list on the last scene; repeat the flag for more lines")
    args = parser.parse_args()

    runs, skipped = load_runs(args.log_dirs)
    if skipped:
        print(f"Skipped {skipped} old-format logs (no per-question data). Re-run to include them.")
    if not runs:
        raise SystemExit("No usable logs found. Run run_groq_reasoning_real_data.py first.")

    # Oracle runs tune on a sample of the new dataset, so their "tuning" numbers measure something
    # else and would mislead on the tuning-versus-new charts. They stay in the text report.
    oracle = [r for r in runs if str(r.get("config", {}).get("tuned_on", "")).startswith("new dataset")]
    if oracle:
        runs = [r for r in runs if r not in oracle]
        print(f"Left {len(oracle)} oracle run(s) out of the page (they tune on the new dataset). They stay in the text report.")
    if not runs:
        raise SystemExit("Only oracle runs found. Add runs from the other optimizers.")

    show = [m.strip() for m in (args.methods or ",".join(DEFAULT_SHOWN)).split(",") if m.strip()]
    if len(show) > 5:
        raise SystemExit(f"The page has room for 5 optimizers; --methods named {len(show)}. Pick 5.")
    keep = set(show) | {"Zero-shot", "Human-written"}
    left_out = sorted({r["name"] for r in runs if r["name"] not in keep})
    runs = [r for r in runs if r["name"] in keep]
    if left_out:
        print("Left out of the page (they are in analyze.py's report): " + ", ".join(left_out) + ". Use --methods to choose which 5 to show.")
    if not any(r["name"] in show for r in runs):
        raise SystemExit("None of the --methods have runs in these logs.")

    sets = group_sets(runs)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": args.title,
        "headline": args.headline,
        "footer": args.footer,
        "next_steps": args.next_step or [],
        "sets": sets,
    }
    for s in sets:
        for r in s["runs"]:
            r.pop("_path", None)

    with open(os.path.join(HERE, "demo_template.html")) as f:
        template = f.read()
    data = json.dumps(payload).replace("</", "<\\/")
    with open(args.out, "w") as f:
        f.write(template.replace("__DATA__", data))

    for s in sets:
        seeds = sorted({r.get("seed", 0) for r in s["runs"]})
        flag = "  (SIMULATED DATA)" if s["simulated"] else ""
        print(f"Run set: {s['label']} | {len(s['runs'])} runs, seeds {seeds}{flag}")
    print(f"Wrote {args.out} ({os.path.getsize(args.out) // 1024} KB). Open it in a browser.")


if __name__ == "__main__":
    main()
