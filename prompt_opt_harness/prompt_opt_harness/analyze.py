#!/usr/bin/env python3
"""Turn a folder of run logs into the project's answer: when do optimized prompts overfit?

Standard library only. Reads the JSON logs written by the run scripts (schema v2) and writes
    results.md   tables, auto-written findings, caveats  (paste-ready for the report)
    results.csv  one row per method (means over seeds)
    fig_gap.svg  dev / source-test / new-dataset accuracy per method

Usage:
    python3 analyze.py --logs logs/after --out analysis/after
    python3 analyze.py --logs logs/after --compare logs/after_1024 --out analysis/after   # token-cap sensitivity

Nothing here calls a model. Every number comes from the logs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
from typing import Dict, List, Optional, Tuple

ORDER = ["Zero-shot", "Human-written", "OPRO", "EvoPrompt-lite", "GEPA-lite", "TextGrad-lite", "Random-search",
         "PBT", "Ensemble", "CostBayesOpt", "Reg-OPRO", "DRO-OPRO", "SAPO-OPRO", "Entropy-OPRO",
         "Guard-OPRO", "Guard-GEPA-lite", "DSPy-GEPA", "DSPy-MIPROv2", "Oracle-OPRO"]
BASELINES = ("Zero-shot", "Human-written")
BOOT = 2000          # bootstrap resamples
TRUNC_OK = 0.02      # <= 2% of answers cut off: fine
TRUNC_BAD = 0.10     # > 10%: results are not trustworthy as they are
SHIFT_BAD = 0.03     # gap moving more than 3 points when cut-off answers are dropped
STABLE_GAP = 0.05    # gap moving more than 5 points when the token cap changes


# ----------------------------------------------------------------------------- loading
def load_runs(log_dir: str) -> Dict[str, Dict[int, dict]]:
    """name -> seed -> payload. Newest file wins when a (method, seed) was run twice. Old logs without
    per-question records (schema < 2) are skipped."""
    found: Dict[Tuple[str, int], Tuple[str, dict]] = {}
    skipped = 0
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            skipped += 1
            continue
        if d.get("schema_version", 1) < 2 or not d.get("eval_records") or "source_test" not in d["eval_records"]:
            skipped += 1
            continue
        key = (d["name"], int(d.get("seed", 0)))
        stamp = d.get("timestamp", "")
        if key not in found or stamp >= found[key][0]:
            found[key] = (stamp, d)
    runs: Dict[str, Dict[int, dict]] = {}
    for (name, seed), (_, d) in found.items():
        runs.setdefault(name, {})[seed] = d
    if skipped:
        print(f"note: skipped {skipped} old or unreadable log file(s) in {log_dir}")
    return runs


def ordered(names) -> List[str]:
    names = list(names)
    return [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)


def is_oracle(d: dict) -> bool:
    return str(d.get("config", {}).get("tuned_on", "")).startswith("new dataset") or d["name"].lower().startswith("oracle")


# ----------------------------------------------------------------------------- small stats
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = list(xs)
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def ms(xs, signed=False, pct=False):
    xs = list(xs)
    if not xs:
        return "n/a"
    m, s = mean(xs), sd(xs)
    if pct:
        return f"{100*m:+.1f} ± {100*s:.1f}" if signed else f"{100*m:.1f} ± {100*s:.1f}"
    return f"{m:+.2f} ± {s:.2f}" if signed else f"{m:.2f} ± {s:.2f}"


def boot_ci(diffs: List[float], rng: random.Random) -> Tuple[float, float, float]:
    n = len(diffs)
    if n == 0:
        return (float("nan"),) * 3
    m = sum(diffs) / n
    means = []
    for _ in range(BOOT):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return m, means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def boot_diff_ci(a: List[float], b: List[float], rng: random.Random) -> Tuple[float, float, float]:
    """CI for mean(a) - mean(b) with the two samples resampled independently."""
    if not a or not b:
        return (float("nan"),) * 3
    na, nb = len(a), len(b)
    out = []
    for _ in range(BOOT):
        out.append(sum(a[rng.randrange(na)] for _ in range(na)) / na - sum(b[rng.randrange(nb)] for _ in range(nb)) / nb)
    out.sort()
    return mean(a) - mean(b), out[int(0.025 * BOOT)], out[int(0.975 * BOOT) - 1]


# ----------------------------------------------------------------------------- per-run views
def records(d: dict) -> Dict[str, List[dict]]:
    return d["eval_records"]


def ood_splits(d: dict) -> List[str]:
    return [k for k in records(d) if k.startswith("ood:")]


def acc(recs: List[dict]) -> float:
    return mean(1.0 if r["correct"] else 0.0 for r in recs)


def run_src_ood(d: dict, drop_truncated: bool = False) -> Tuple[float, float]:
    """(source-test accuracy, mean accuracy over new datasets) for one run."""
    def f(recs):
        if drop_truncated:
            recs = [r for r in recs if not r.get("truncated")]
        return acc(recs) if recs else float("nan")
    src = f(records(d)["source_test"])
    oods = [f(records(d)[k]) for k in ood_splits(d)]
    oods = [o for o in oods if o == o]
    return src, (mean(oods) if oods else float("nan"))


def pooled(runs_of_method: Dict[int, dict], split_filter) -> Dict[tuple, float]:
    """(seed, split, id) -> 1.0/0.0 for every question in the splits that pass split_filter."""
    out = {}
    for seed, d in runs_of_method.items():
        for split, recs in records(d).items():
            if split_filter(split):
                for r in recs:
                    out[(seed, split, r["id"])] = 1.0 if r["correct"] else 0.0
    return out


# ----------------------------------------------------------------------------- truncation audit
def truncation_rows(runs) -> Tuple[List[List[str]], dict]:
    rows, summary = [], {"max_rate": 0.0, "max_shift": 0.0, "worst_method": None, "flagged": []}
    for name in ordered(runs):
        rs = list(runs[name].values())
        src = [r for d in rs for r in records(d)["source_test"]]
        ood = [r for d in rs for k in ood_splits(d) for r in records(d)[k]]
        allr = src + ood
        if not allr:
            continue
        t = lambda xs: mean(1.0 if r.get("truncated") else 0.0 for r in xs) if xs else float("nan")
        unf = mean(0.0 if r.get("well_formed", True) else 1.0 for r in allr)
        toks = mean(r.get("output_tokens") or 0 for r in allr)
        gap_all = mean(run_src_ood(d)[0] - run_src_ood(d)[1] for d in rs)
        gap_ex = mean(run_src_ood(d, True)[0] - run_src_ood(d, True)[1] for d in rs)
        shift = abs(gap_all - gap_ex) if gap_ex == gap_ex else float("nan")
        rate = t(allr)
        if rate > summary["max_rate"]:
            summary["max_rate"], summary["worst_method"] = rate, name
        if shift == shift:
            summary["max_shift"] = max(summary["max_shift"], shift)
        if rate > TRUNC_OK or (shift == shift and shift > SHIFT_BAD):
            summary["flagged"].append(name)
        rows.append([name, str(len(allr)), f"{100*rate:.1f}%", f"{100*t(src):.1f}%", f"{100*t(ood):.1f}%",
                     f"{100*unf:.1f}%", f"{toks:.0f}", f"{gap_all:+.3f}",
                     f"{gap_ex:+.3f}" if gap_ex == gap_ex else "n/a"])
    r, s = summary["max_rate"], summary["max_shift"]
    if r > TRUNC_BAD or s > SHIFT_BAD:
        summary["verdict"] = "CONFOUNDED"
        summary["text"] = (f"Cut-off answers are a real problem: up to {100*r:.1f}% of answers were truncated "
                           f"({summary['worst_method']}), and dropping them moves a gap by up to {s:.3f}. "
                           "Rerun with a larger cap (`MAX_TOKENS=1024 LOG_DIR=logs/after_1024 bash run_all.sh`) before drawing conclusions.")
    elif r > TRUNC_OK:
        summary["verdict"] = "CHECK"
        summary["text"] = (f"Some answers were cut off (up to {100*r:.1f}%, {summary['worst_method']}), but dropping them moves no gap by more than "
                           f"{s:.3f}. Probably not driving the results; confirm with a rerun at a larger cap.")
    else:
        summary["verdict"] = "OK"
        summary["text"] = (f"Truncation is not driving the results: at most {100*r:.1f}% of answers were cut off for any method, "
                           f"and dropping them moves no gap by more than {s:.3f}.")
    return rows, summary


# ----------------------------------------------------------------------------- comparisons vs a reference method
def compare(runs, ref: str, rng: random.Random) -> List[dict]:
    if ref not in runs:
        return []
    ref_src = pooled(runs[ref], lambda s: s == "source_test")
    ref_ood = pooled(runs[ref], lambda s: s.startswith("ood:"))
    out = []
    for name in ordered(runs):
        if name == ref:
            continue
        m_src = pooled(runs[name], lambda s: s == "source_test")
        m_ood = pooled(runs[name], lambda s: s.startswith("ood:"))
        ds = [m_src[k] - ref_src[k] for k in m_src if k in ref_src]
        do = [m_ood[k] - ref_ood[k] for k in m_ood if k in ref_ood]
        g_src, s_lo, s_hi = boot_ci(ds, rng)
        g_ood, o_lo, o_hi = boot_ci(do, rng)
        # transfer shortfall: how much more the prompt gained on the source than on the new data
        short, sh_lo, sh_hi = boot_diff_ci(ds, do, rng)
        oracle = any(is_oracle(d) for d in runs[name].values())
        if name in BASELINES:
            label = "baseline"
        elif oracle:
            label = "oracle (tuned on the new data; upper bound)"
        elif s_lo > 0 and sh_lo > 0 and o_lo > 0:
            label = "partly transfers: gain reliably smaller on new data"
        elif s_lo > 0 and sh_lo > 0:
            label = "OVERFITS: gain on source is reliably larger than on new data"
        elif s_lo > 0 and o_lo > 0:
            label = "gain transfers"
        elif s_lo > 0:
            label = "gain on source; transfer inconclusive (need more questions)"
        elif o_hi < 0:
            label = "hurts on new data"
        elif s_hi < 0:
            label = "worse than the reference on source"
        else:
            label = "no reliable gain on source (nothing to overfit)"
        out.append(dict(name=name, n_src=len(ds), n_ood=len(do), g_src=g_src, s_lo=s_lo, s_hi=s_hi,
                        g_ood=g_ood, o_lo=o_lo, o_hi=o_hi, short=short, sh_lo=sh_lo, sh_hi=sh_hi, label=label, oracle=oracle))
    return out


def pp(x):  # percentage points
    return "n/a" if x != x else f"{100*x:+.1f}"


def cmp_table(rows: List[dict]) -> str:
    head = "| Method | Gain on source (pts) | 95% CI | Gain on new data (pts) | 95% CI | Transfer shortfall (pts) | 95% CI | Reading |\n|---|---|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        body += (f"| {r['name']} | {pp(r['g_src'])} | [{pp(r['s_lo'])}, {pp(r['s_hi'])}] | {pp(r['g_ood'])} | "
                 f"[{pp(r['o_lo'])}, {pp(r['o_hi'])}] | {pp(r['short'])} | [{pp(r['sh_lo'])}, {pp(r['sh_hi'])}] | {r['label']} |\n")
    return head + body


# ----------------------------------------------------------------------------- main metrics table
def metric_rows(runs) -> Tuple[List[List[str]], List[dict]]:
    rows, flat = [], []
    zs = runs.get("Zero-shot", {})
    for name in ordered(runs):
        rs = [runs[name][s] for s in sorted(runs[name])]
        dev = [d["dev_score"] for d in rs]
        src = [d["source_test_acc"] for d in rs]
        gap = [d["dev_minus_test"] for d in rs]
        ood = [run_src_ood(d)[1] for d in rs]
        og = [d["ood_gap"] for d in rs]
        worst = [d["worst_case_acc"] for d in rs]
        var = [d["variance"] for d in rs]
        calls = [d["api_calls"] for d in rs]
        toks = [(d.get("input_tokens", 0) or 0) + (d.get("output_tokens", 0) or 0) for d in rs]
        gain_per_call = float("nan")
        if zs and name != "Zero-shot":
            gains = [d["source_test_acc"] - zs[s]["source_test_acc"] for s, d in runs[name].items() if s in zs]
            if gains and mean(calls) > 0:
                gain_per_call = mean(gains) / mean(calls)
        rows.append([name, str(len(rs)), ms(dev), ms(src), ms(gap, True), ms(ood), ms(og, True), ms(worst),
                     f"{mean(var):.4f}", f"{mean(calls):.0f}", f"{mean(toks)/1000:.0f}k",
                     "n/a" if gain_per_call != gain_per_call else f"{1000*gain_per_call:+.2f}"])
        flat.append(dict(method=name, seeds=len(rs), dev=mean(dev), source_test=mean(src), dev_minus_test=mean(gap),
                         new_data_acc=mean(ood), ood_gap=mean(og), worst_case=mean(worst), cross_domain_variance=mean(var),
                         api_calls=mean(calls), tokens=mean(toks), gain_per_call=gain_per_call))
    return rows, flat


def per_domain_rows(runs) -> Tuple[List[str], List[List[str]]]:
    """Accuracy on each new dataset (mean ± sd over seeds) and the spread between them (Q2: robustness)."""
    domains = sorted({k[4:] for rs in runs.values() for d in rs.values() for k in ood_splits(d)})
    rows = []
    for name in ordered(runs):
        rs = [runs[name][s] for s in sorted(runs[name])]
        cells = []
        per = []
        for dom in domains:
            vals = [acc(records(d)["ood:" + dom]) for d in rs if "ood:" + dom in records(d)]
            cells.append(ms(vals) if vals else "n/a")
            if vals:
                per.append(mean(vals))
        spread = f"{max(per) - min(per):.2f}" if len(per) > 1 else "n/a (one dataset)"
        rows.append([name] + cells + [f"{min(per):.2f}" if per else "n/a", spread])
    return ["Method"] + [f"{d}" for d in domains] + ["Worst dataset", "Spread between datasets"], rows


def guard_rows(runs) -> List[List[str]]:
    """What the confirmation guard decided for each guarded method (kept the search winner or fell back)."""
    rows = []
    for name in ordered(runs):
        gs = [d.get("extra", {}).get("guard") for d in runs[name].values()]
        gs = [g for g in gs if g]
        if not gs:
            continue
        checked = [g for g in gs if g.get("checked")]
        kept = [g for g in checked if g["kept"] == "search winner"]
        rows.append([name, str(len(gs)), str(len(gs) - len(checked)), str(len(checked)), str(len(kept)),
                     f"{mean(g['start_confirm_acc'] for g in checked):.2f}" if checked else "n/a",
                     f"{mean(g['winner_confirm_acc'] for g in checked):.2f}" if checked else "n/a"])
    return rows


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out


# ----------------------------------------------------------------------------- token-cap sensitivity
def sensitivity(runs_a, runs_b, cap_a, cap_b) -> Tuple[str, str, bool]:
    rows, stable = [], True
    for name in ordered(set(runs_a) & set(runs_b)):
        sa = [run_src_ood(d) for d in runs_a[name].values()]
        sb = [run_src_ood(d) for d in runs_b[name].values()]
        ga = mean(a - b for a, b in sa)
        gb = mean(a - b for a, b in sb)
        ta = mean(1.0 if r.get("truncated") else 0.0 for d in runs_a[name].values() for rs in records(d).values() for r in rs)
        tb = mean(1.0 if r.get("truncated") else 0.0 for d in runs_b[name].values() for rs in records(d).values() for r in rs)
        if abs(ga - gb) > STABLE_GAP:
            stable = False
        rows.append([name, f"{100*ta:.1f}% → {100*tb:.1f}%", f"{mean(a for a,_ in sa):.2f} → {mean(a for a,_ in sb):.2f}",
                     f"{mean(b for _,b in sa):.2f} → {mean(b for _,b in sb):.2f}", f"{ga:+.2f} → {gb:+.2f}", f"{gb-ga:+.2f}"])
    table = md_table(["Method", "Truncated", "Source test", "New data", "Source − new gap", "Gap change"], rows)
    if not rows:
        return "", "No methods appear in both folders, so the token cap could not be compared.", False
    text = (f"Raising the cap from {cap_a} to {cap_b} tokens moves no gap by more than {STABLE_GAP:.2f}: the conclusions do not depend on the cap."
            if stable else
            f"Raising the cap from {cap_a} to {cap_b} tokens changes at least one gap by more than {STABLE_GAP:.2f}. "
            f"Report the {cap_b}-token numbers and treat the {cap_a}-token ones as truncation-affected.")
    text += " (The search is re-run at each cap, so part of any change is ordinary search noise, not only the cap.)"
    return table, text, stable


# ----------------------------------------------------------------------------- figure
def svg_figure(flat: List[dict], title: str) -> str:
    rows = [f for f in flat]
    W, rowh, left, right, top = 760, 34, 150, 30, 70
    H = top + rowh * len(rows) + 50
    vals = [v for f in rows for v in (f["dev"], f["source_test"], f["new_data_acc"]) if v == v]
    lo = max(0.0, min(vals) - 0.08) if vals else 0.0
    hi = min(1.0, max(vals) + 0.08) if vals else 1.0
    x = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    c_dev, c_src, c_new, c_line, c_txt, c_grid = "#8a8f98", "#2f6fb0", "#d9622b", "#b9bec6", "#1f2328", "#e4e7eb"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{left}" y="24" font-size="15" font-weight="bold" fill="{c_txt}">{title}</text>',
         f'<text x="{left}" y="42" fill="#555">Mean over seeds. The line is the change from the source test to the new dataset.</text>']
    t = 0.0
    ticks = []
    v = int(lo * 10) / 10
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(v)
        v = round(v + 0.1, 10)
    for tv in ticks:
        o.append(f'<line x1="{x(tv):.1f}" y1="{top-12}" x2="{x(tv):.1f}" y2="{top+rowh*len(rows)-8}" stroke="{c_grid}"/>')
        o.append(f'<text x="{x(tv):.1f}" y="{top+rowh*len(rows)+6}" text-anchor="middle" fill="#666">{int(round(tv*100))}%</text>')
    for i, f in enumerate(rows):
        y = top + i * rowh + 6
        o.append(f'<text x="{left-12}" y="{y+4}" text-anchor="end" fill="{c_txt}">{f["method"]}</text>')
        if f["source_test"] == f["source_test"] and f["new_data_acc"] == f["new_data_acc"]:
            o.append(f'<line x1="{x(f["source_test"]):.1f}" y1="{y}" x2="{x(f["new_data_acc"]):.1f}" y2="{y}" stroke="{c_line}" stroke-width="3"/>')
        o.append(f'<circle cx="{x(f["dev"]):.1f}" cy="{y}" r="5" fill="#fff" stroke="{c_dev}" stroke-width="2"/>')
        o.append(f'<circle cx="{x(f["source_test"]):.1f}" cy="{y}" r="6" fill="{c_src}"/>')
        if f["new_data_acc"] == f["new_data_acc"]:
            cx, r = x(f["new_data_acc"]), 6
            o.append(f'<polygon points="{cx:.1f},{y-r-1} {cx+r+1:.1f},{y} {cx:.1f},{y+r+1} {cx-r-1:.1f},{y}" fill="{c_new}"/>')
    ly = H - 16
    items = [("Dev (what the optimizer saw)", c_dev, "o"), ("Source test (same dataset)", c_src, "c"), ("New dataset", c_new, "d")]
    lx = left
    for text, col, shape in items:
        if shape == "o":
            o.append(f'<circle cx="{lx}" cy="{ly-4}" r="5" fill="#fff" stroke="{col}" stroke-width="2"/>')
        elif shape == "c":
            o.append(f'<circle cx="{lx}" cy="{ly-4}" r="6" fill="{col}"/>')
        else:
            o.append(f'<polygon points="{lx},{ly-11} {lx+7},{ly-4} {lx},{ly+3} {lx-7},{ly-4}" fill="{col}"/>')
        o.append(f'<text x="{lx+12}" y="{ly}" fill="{c_txt}">{text}</text>')
        lx += 12 + 8 * len(text) + 28
    o.append("</svg>")
    return "\n".join(o)


# ----------------------------------------------------------------------------- findings
def findings(runs, flat, cmp_zs, cmp_hw, trunc) -> List[str]:
    out = [f"**Token cap:** {trunc['text']}"]
    real = [r for r in cmp_zs if not r["oracle"] and r["name"] not in BASELINES]
    if real:
        over = [r["name"] for r in real if r["label"].startswith("OVERFITS")]
        trans = [r["name"] for r in real if r["label"] == "gain transfers"]
        part = [r["name"] for r in real if r["label"].startswith("partly")]
        inc = [r["name"] for r in real if r["label"].startswith("gain on source; transfer inconclusive")]
        none = [r["name"] for r in real if r["label"].startswith("no reliable")]
        hurt = [r["name"] for r in real if r["label"].startswith("hurts") or r["label"].startswith("worse")]
        bits = []
        if trans:
            bits.append(f"gains that transfer to the new dataset: {', '.join(trans)}")
        if part:
            bits.append(f"gains that shrink reliably on the new dataset: {', '.join(part)}")
        if over:
            bits.append(f"gains on the source that are reliably larger than on the new dataset (overfitting): {', '.join(over)}")
        if inc:
            bits.append(f"gains on the source where the data cannot yet say whether they transfer: {', '.join(inc)}")
        if none:
            bits.append(f"no reliable gain over Zero-shot even on the source: {', '.join(none)}")
        if hurt:
            bits.append(f"worse than Zero-shot: {', '.join(hurt)}")
        out.append("**Versus Zero-shot (95% bootstrap CIs over questions):** " + "; ".join(bits) + ".")
    hw = [r for r in cmp_hw if not r["oracle"] and r["name"] not in BASELINES]
    if hw:
        beat = [r["name"] for r in hw if r["g_ood"] == r["g_ood"] and r["o_lo"] > 0]
        lose = [r["name"] for r in hw if r["g_ood"] == r["g_ood"] and r["o_hi"] < 0]
        s = "No optimizer reliably beats the human-written prompt on the new dataset."
        if beat or lose:
            s = ""
            if beat:
                s += f"Beats the human-written prompt on the new dataset: {', '.join(beat)}. "
            if lose:
                s += f"Reliably worse than the human-written prompt on the new dataset: {', '.join(lose)}. "
        out.append("**Versus the human-written prompt:** " + s.strip())
    opt = [f for f in flat if f["method"] not in BASELINES and not any(is_oracle(d) for d in runs[f["method"]].values())]
    if opt:
        big = max(opt, key=lambda f: f["dev_minus_test"])
        rob = max(opt, key=lambda f: f["worst_case"])
        out.append(f"**Dev optimism:** the largest dev − test gap is {big['method']} ({big['dev_minus_test']:+.2f}); "
                   "a large value means the search picked a prompt that looked better on the dev sample than it really is.")
        out.append(f"**Most robust:** {rob['method']} has the best worst-case accuracy over the new datasets ({rob['worst_case']:.2f}).")
        orc = next((f for f in flat if any(is_oracle(d) for d in runs[f["method"]].values())), None)
        if orc and orc["new_data_acc"] == orc["new_data_acc"]:
            best = max(opt, key=lambda f: f["new_data_acc"])
            head = orc["new_data_acc"] - best["new_data_acc"]
            if head > 0.005:
                out.append(f"**Headroom:** tuning on the new dataset (oracle) reaches {orc['new_data_acc']:.2f} on it, versus {best['new_data_acc']:.2f} for the best "
                           f"source-tuned optimizer ({best['method']}): {100*head:.1f} points left on the table by not seeing the target domain.")
            else:
                out.append(f"**Headroom:** tuning on the new dataset (oracle) reaches {orc['new_data_acc']:.2f} on it, no better than {best['new_data_acc']:.2f} for the best "
                           f"source-tuned optimizer ({best['method']}): seeing the target domain did not help here.")
        cheap = [f for f in opt if f["gain_per_call"] == f["gain_per_call"] and f["gain_per_call"] > 0]
        if not cheap:
            out.append("**Cost:** no optimizer improved source-test accuracy over Zero-shot, so there is no gain per API call to report.")
        else:
            c = max(cheap, key=lambda f: f["gain_per_call"])
            out.append(f"**Cost:** best gain per API call (source test vs Zero-shot) is {c['method']} ({1000*c['gain_per_call']:+.2f} points per 1000 calls).")
    return out


# ----------------------------------------------------------------------------- report
def build(args) -> str:
    runs = load_runs(args.logs)
    if not runs:
        raise SystemExit(f"No usable logs in {args.logs}. Run the experiment first (see run_all.sh).")
    rng = random.Random(0)
    cfgs = [d.get("config", {}) for rs in runs.values() for d in rs.values()]
    cap = cfgs[0].get("max_tokens", "?")
    model = cfgs[0].get("model", "?")
    seeds = sorted({s for rs in runs.values() for s in rs})
    n_src = len(next(iter(next(iter(runs.values())).values()))["eval_records"]["source_test"])
    n_new = sum(len(v) for k, v in next(iter(next(iter(runs.values())).values()))["eval_records"].items() if k.startswith("ood:"))

    rows, flat = metric_rows(runs)
    trows, tsum = truncation_rows(runs)
    cz, ch = compare(runs, "Zero-shot", rng), compare(runs, "Human-written", rng)
    sens = None
    if args.compare:
        other = load_runs(args.compare)
        ocap = next(iter(next(iter(other.values())).values())).get("config", {}).get("max_tokens", "?") if other else "?"
        sens = sensitivity(runs, other, cap, ocap) + (ocap,)
        tsum["text"] += " Token-cap comparison: " + sens[1]

    md = [f"# Results: when do optimized prompts overfit?\n",
          f"Model `{model}`, max_tokens {cap}, seeds {seeds}, {n_src} source-test and {n_new} new-dataset questions per seed. "
          f"Optimizers were tuned on the source dev sample only (the oracle excepted).\n",
          "## Findings (auto-generated from the logs)\n"]
    md += [f"- {f}" for f in findings(runs, flat, cz, ch, tsum)]
    md += ["", "## 1. Headline metrics (mean ± sd over seeds)\n",
           md_table(["Method", "Seeds", "Dev", "Source test", "Dev − test", "New data", "Source − new gap", "Worst case", "Variance",
                     "API calls", "Tokens", "Gain / 1000 calls"], rows),
           "Dev − test and Source − new gap are signed: positive means accuracy fell. Worst case is the lowest accuracy over the new datasets. "
           "Gain / 1000 calls is the source-test gain over Zero-shot per 1000 API calls (all calls, search included).\n",
           "## 2. Do the gains transfer? Paired comparison with Zero-shot\n", cmp_table(cz),
           "Gains are in percentage points, paired by question and pooled over seeds; CIs resample questions. "
           "Transfer shortfall = gain on source − gain on new data; positive means the improvement did not carry over. "
           "OVERFITS requires a reliably positive source gain and a reliably positive transfer shortfall (both CIs above 0); a gain whose shortfall CI includes 0 is reported as inconclusive, not as overfitting.\n"]
    if ch:
        md += ["## 3. Against the human-written prompt\n", cmp_table(ch)]
    ph, prow = per_domain_rows(runs)
    md += ["## 4. Robustness across new datasets (Q2)\n", md_table(ph, prow),
           "Accuracy per new dataset, mean ± sd over seeds. Spread = best minus worst dataset; a small spread means the prompt holds up evenly across datasets. "
           "With one new dataset there is no spread to measure: run with two or more (`OOD=svamp,gsmplus`).\n"]
    grows = guard_rows(runs)
    if grows:
        md += ["## 5. Confirmation guard decisions\n",
               md_table(["Method", "Runs", "Search kept starting prompt", "Winner checked", "Winner kept", "Starting prompt on confirm set", "Winner on confirm set"], grows),
               "The guard keeps the search winner only if it beats the starting prompt on separate source questions the search never saw; ties and losses keep the starting prompt.\n"]
    md += ["## 6. Truncation audit (is the 512-token cap the confound?)\n",
           md_table(["Method", "Answers", "Truncated", "Source", "New data", "Unparseable", "Mean output tokens", "Gap (all)", "Gap (cut-off dropped)"], trows),
           f"**Verdict: {tsum['verdict']}.** {tsum['text']}\n"]
    if sens:
        table, text, stable, ocap = sens
        md += [f"## 7. Token-cap sensitivity ({cap} vs {ocap})\n", table, text + "\n"]
    zs0 = next((f for f in flat if f["method"] == "Zero-shot"), None)
    ceiling = []
    if zs0 and max(zs0["source_test"], zs0["new_data_acc"]) >= 0.90:
        ceiling = [f"- Near ceiling: Zero-shot already scores {zs0['source_test']:.2f} on the source and {zs0['new_data_acc']:.2f} on the new data, "
                   "so a 3-point difference is only 1-2 questions out of a few dozen. Treat small gains as noise unless the CI excludes 0."]
    many = []
    n_compared = len([r for r in cz if not r["oracle"] and r["name"] not in BASELINES])
    if n_compared > 6:
        many = [f"- {n_compared} optimizers are compared with 95% intervals, so about 1 in 20 comparisons will look significant by chance. "
                "Treat a single win among many as a lead to confirm, not a conclusion."]
    md += ["## Caveats\n"] + ceiling + many + [
           f"- Question-level CIs are wide with {n_src} source and {n_new} new-data questions per seed; a method listed as having no reliable gain may still have a small real one. "
           "Raise `N_PER_SPLIT` to tighten them.",
           "- One model, one task family (math word problems), one source → target shift (GSM8K → SVAMP). The conclusions are about this setting.",
           "- Seeds change the question sample and the optimizer's proposal sampling, so seed spread mixes data noise and search noise.",
           "- GEPA-lite and TextGrad-lite are simplified re-implementations; DSPy-GEPA and DSPy-MIPROv2 are the real DSPy optimizers, restricted to instruction-only search through this harness."]
    text = "\n".join(md) + "\n"

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    with open(os.path.join(args.out, "fig_gap.svg"), "w") as f:
        f.write(svg_figure(flat, f"Overfitting by method: {model}"))
    if not getattr(args, "quiet", False):
        print(text)
        print(f"Wrote {args.out}/results.md, results.csv, fig_gap.svg")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, help="folder of run logs, e.g. logs/after")
    ap.add_argument("--out", default="analysis", help="where to write results.md / results.csv / fig_gap.svg")
    ap.add_argument("--quiet", action="store_true", help="write the files without printing the report")
    ap.add_argument("--compare", default=None, help="a second log folder run with a different --max-tokens, for the sensitivity table")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
