"""Categorize why SWE-bench instances failed, instead of just counting them.

`resolved: false` is a single bit. This tool joins three sources to say *how* an
instance failed, which is what tells you whether to fix the model, the prompt, or
the turn budget:

  predictions.jsonl  -> did the agent error out? produce no diff? hit its turn cap?
  report.json        -> which FAIL_TO_PASS tests still fail
  dataset gold patch -> did the agent even touch the files the real fix touched?

That last join is the useful one: "never opened the right file" (a localization
failure) and "edited the right file but got the logic wrong" (a reasoning failure)
are different problems with different fixes.

Usage:
    python -m evals.swebench.analyze_failures \
        --predictions evals/swebench/out/predictions_kimi50.jsonl --run-id kimi-50
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Lite"
SPLIT = "test"
DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def patched_files(patch: str) -> set[str]:
    """Paths a unified diff touches."""
    return {m.group(1) for m in DIFF_FILE_RE.finditer(patch or "")}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_reports(logs_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in logs_dir.glob("*/report.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        iid, body = next(iter(data.items()))
        out[iid] = body
    return out


def classify(pred: dict, report: dict | None, gold_files: set[str]) -> tuple[str, str]:
    """-> (category, detail). Order matters: earlier checks are more specific."""
    if pred.get("_error"):
        return "agent_error", str(pred["_error"])[:120]
    if pred.get("_empty_patch"):
        return "empty_patch", f"{pred.get('_llm_calls', 0)} calls, no diff produced"
    if report is None:
        return "not_scored", "no report.json"
    if not report.get("patch_successfully_applied", False):
        return "patch_did_not_apply", "harness could not apply the diff"

    agent_files = patched_files(pred.get("model_patch", ""))
    overlap = agent_files & gold_files
    f2p_fail = report.get("tests_status", {}).get("FAIL_TO_PASS", {}).get("failure", [])
    p2p_fail = report.get("tests_status", {}).get("PASS_TO_PASS", {}).get("failure", [])

    if p2p_fail:
        return "regression", f"broke {len(p2p_fail)} previously-passing test(s)"
    if not overlap:
        return "wrong_file", (
            f"touched {sorted(agent_files) or '[]'}; gold touched {sorted(gold_files)}"
        )
    return "wrong_logic", (
        f"right file(s) {sorted(overlap)}, but {len(f2p_fail)} FAIL_TO_PASS still failing"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evals.swebench.analyze_failures")
    p.add_argument("--predictions", required=True)
    p.add_argument("--run-id", help="Run id under ~/logs/run_evaluation/.")
    p.add_argument("--logs-dir", help="Explicit logs dir (overrides --run-id).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    preds = {d["instance_id"]: d for d in load_jsonl(Path(os.path.expanduser(args.predictions)))}
    logs = Path(os.path.expanduser(args.logs_dir)) if args.logs_dir else (
        Path(os.path.expanduser("~")) / "logs" / "run_evaluation" / args.run_id
    )
    # reports live one level down, under the model name
    sub = [d for d in logs.iterdir() if d.is_dir()] if logs.exists() else []
    reports: dict[str, dict] = {}
    for d in sub:
        reports.update(load_reports(d))
    if not reports:
        reports = load_reports(logs)

    from datasets import load_dataset
    gold = {
        r["instance_id"]: patched_files(r["patch"])
        for r in load_dataset(DATASET, split=SPLIT)
        if r["instance_id"] in preds
    }

    rows = []
    for iid, pred in sorted(preds.items()):
        rep = reports.get(iid)
        if rep is not None and rep.get("resolved"):
            continue  # only analyse failures
        cat, detail = classify(pred, rep, gold.get(iid, set()))
        rows.append({
            "instance_id": iid,
            "category": cat,
            "detail": detail,
            "llm_calls": pred.get("_llm_calls"),
            "hit_turn_cap": pred.get("_hit_turn_cap"),
            "input_tokens": pred.get("_input_tokens"),
        })

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    print(f"failures: {len(rows)} of {len(preds)} instances\n")
    print("by category:")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<22} {n}")
    print()
    for r in rows:
        cap = " [HIT TURN CAP]" if r["hit_turn_cap"] else ""
        print(f"- {r['instance_id']}  [{r['category']}]{cap}")
        print(f"    calls={r['llm_calls']} in_tok={r['input_tokens']}")
        print(f"    {r['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
