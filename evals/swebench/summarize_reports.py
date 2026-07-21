"""Summarize SWE-bench per-instance eval reports into a resolved-rate table.

The official harness writes a per-instance `report.json` under
`logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json` *before* it
builds the final summary. On networks where the summary step fails (e.g. it
fetches every repo's requirements from raw.githubusercontent.com, which is
unreliable behind the GFW — see README), those per-instance files are still
valid. This tool reads them directly so you always get a score.

Usage:
    python -m evals.swebench.summarize_reports --run-id agent-3
    python -m evals.swebench.summarize_reports --logs-dir ~/logs/run_evaluation/agent-3
    python -m evals.swebench.summarize_reports --run-id agent-10 --json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def find_reports(logs_dir: Path) -> list[Path]:
    """All per-instance report.json files under a run's logs dir."""
    return sorted(logs_dir.glob("*/*/report.json")) or sorted(logs_dir.glob("*/report.json"))


def load_report(path: Path) -> dict:
    """A report.json is {instance_id: {...}}; flatten to the inner dict + id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    instance_id, body = next(iter(data.items()))
    body = dict(body)
    body["instance_id"] = instance_id
    return body


def summarize(logs_dir: Path) -> dict:
    reports = [load_report(p) for p in find_reports(logs_dir)]
    resolved = [r for r in reports if r.get("resolved")]
    applied = [r for r in reports if r.get("patch_successfully_applied")]
    return {
        "logs_dir": str(logs_dir),
        "total": len(reports),
        "resolved": len(resolved),
        "resolved_rate": round(len(resolved) / len(reports), 4) if reports else 0.0,
        "patch_applied": len(applied),
        "instances": [
            {
                "instance_id": r["instance_id"],
                "resolved": bool(r.get("resolved")),
                "patch_applied": bool(r.get("patch_successfully_applied")),
            }
            for r in sorted(reports, key=lambda r: r["instance_id"])
        ],
    }


def _default_logs_dir(run_id: str) -> Path:
    return Path(os.path.expanduser("~")) / "logs" / "run_evaluation" / run_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evals.swebench.summarize_reports")
    p.add_argument("--run-id", help="Run id under ~/logs/run_evaluation/.")
    p.add_argument("--logs-dir", help="Explicit path to a run's logs dir (overrides --run-id).")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = p.parse_args(argv)

    if args.logs_dir:
        logs_dir = Path(os.path.expanduser(args.logs_dir))
    elif args.run_id:
        logs_dir = _default_logs_dir(args.run_id)
    else:
        p.error("pass --run-id or --logs-dir")

    if not logs_dir.exists():
        raise SystemExit(f"logs dir not found: {logs_dir}")

    result = summarize(logs_dir)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"logs: {result['logs_dir']}")
    print(f"{'instance_id':<40} {'applied':>8} {'resolved':>9}")
    print("-" * 60)
    for it in result["instances"]:
        print(f"{it['instance_id']:<40} {str(it['patch_applied']):>8} {str(it['resolved']):>9}")
    print("-" * 60)
    print(
        f"resolved {result['resolved']}/{result['total']} "
        f"({result['resolved_rate'] * 100:.1f}%)  |  patch applied {result['patch_applied']}/{result['total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
