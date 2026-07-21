"""Summarize SWE-bench per-instance eval reports into a resolved-rate table.

The official harness writes a per-instance `report.json` under
`logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json` *before* it
builds the final summary. On networks where the summary step fails (e.g. it
fetches every repo's requirements from raw.githubusercontent.com, which is
unreliable behind the GFW — see README), those per-instance files are still
valid. This tool reads them directly so you always get a score.

Pass `--predictions predictions.jsonl` to join in the generation-side
diagnostics the adapter records (tokens, LLM calls, wall time, whether the run
hit its turn budget). A resolved rate alone hides efficiency: two agents can
both score 50% while one burns 10x the tokens. With `--price-in/--price-out`
(USD per 1M tokens) it also reports cost and cost-per-resolved.

Usage:
    python -m evals.swebench.summarize_reports --run-id agent-10
    python -m evals.swebench.summarize_reports --run-id agent-10 \
        --predictions evals/swebench/out/predictions_10.jsonl
    python -m evals.swebench.summarize_reports --run-id agent-10 \
        --predictions evals/swebench/out/predictions_10.jsonl \
        --price-in 0.3 --price-out 1.2 --json
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


def load_predictions(path: Path) -> dict[str, dict]:
    """Map instance_id -> the adapter's generation diagnostics."""
    preds: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        preds[p["instance_id"]] = p
    return preds


def _cost(in_tok: int, out_tok: int, price_in: float | None, price_out: float | None) -> float | None:
    if price_in is None or price_out is None:
        return None
    return in_tok / 1e6 * price_in + out_tok / 1e6 * price_out


def summarize(
    logs_dir: Path,
    predictions: dict[str, dict] | None = None,
    *,
    price_in: float | None = None,
    price_out: float | None = None,
    max_turns: int | None = None,
) -> dict:
    predictions = predictions or {}
    reports = {r["instance_id"]: r for r in (load_report(p) for p in find_reports(logs_dir))}

    # An instance whose patch was empty never gets a report — the harness has
    # nothing to apply. Those still count as unresolved in the SWE-bench metric,
    # so when we know the full prediction set, iterate over *it*, not the reports;
    # otherwise the rate is silently computed over a shrunken denominator.
    ids = sorted(predictions) if predictions else sorted(reports)

    instances = []
    for iid in ids:
        r = reports.get(iid)
        row = {
            "instance_id": iid,
            "resolved": bool(r.get("resolved")) if r else False,
            "patch_applied": bool(r.get("patch_successfully_applied")) if r else False,
            "evaluated": r is not None,
        }
        pred = predictions.get(iid)
        if pred is not None:
            row["empty_patch"] = bool(pred.get("_empty_patch"))
            in_tok = pred.get("_input_tokens") or 0
            out_tok = pred.get("_output_tokens") or 0
            # prefer the recorded flag; fall back to inferring from calls vs --max-turns
            hit_cap = pred.get("_hit_turn_cap")
            if hit_cap is None and max_turns is not None:
                hit_cap = (pred.get("_llm_calls") or 0) >= max_turns
            row.update(
                input_tokens=in_tok,
                output_tokens=out_tok,
                llm_calls=pred.get("_llm_calls") or 0,
                duration_s=pred.get("_duration_s") or 0.0,
                hit_turn_cap=hit_cap,
                cost=_cost(in_tok, out_tok, price_in, price_out),
            )
        instances.append(row)

    resolved = [it for it in instances if it["resolved"]]
    applied = [it for it in instances if it["patch_applied"]]

    def _agg(rows: list[dict]) -> dict:
        n = len(rows)
        if not n or "input_tokens" not in rows[0]:
            return {"n": n}
        tin = sum(r["input_tokens"] for r in rows)
        tout = sum(r["output_tokens"] for r in rows)
        costs = [r["cost"] for r in rows if r.get("cost") is not None]
        return {
            "n": n,
            "input_tokens": tin,
            "output_tokens": tout,
            "avg_input_tokens": round(tin / n),
            "avg_output_tokens": round(tout / n),
            "avg_calls": round(sum(r["llm_calls"] for r in rows) / n, 1),
            "avg_duration_s": round(sum(r["duration_s"] for r in rows) / n, 1),
            "cost": round(sum(costs), 4) if costs else None,
        }

    evaluated = [it for it in instances if it["evaluated"]]
    empty = [it for it in instances if it.get("empty_patch")]

    return {
        "logs_dir": str(logs_dir),
        "total": len(instances),
        "resolved": len(resolved),
        "resolved_rate": round(len(resolved) / len(instances), 4) if instances else 0.0,
        # rate among instances that actually produced a patch and got scored
        "evaluated": len(evaluated),
        "empty_patches": len(empty),
        "resolved_rate_of_evaluated": (
            round(len(resolved) / len(evaluated), 4) if evaluated else 0.0
        ),
        "patch_applied": len(applied),
        "has_predictions": bool(predictions),
        "overall": _agg(instances),
        "resolved_stats": _agg(resolved),
        "unresolved_stats": _agg([it for it in instances if not it["resolved"]]),
        "instances": instances,
    }


def _default_logs_dir(run_id: str) -> Path:
    return Path(os.path.expanduser("~")) / "logs" / "run_evaluation" / run_id


def _fmt_bool(v) -> str:
    return "-" if v is None else str(v)


def _print_table(result: dict) -> None:
    print(f"logs: {result['logs_dir']}")
    withp = result["has_predictions"]
    has_cost = any(it.get("cost") is not None for it in result["instances"])
    if withp:
        hdr = f"{'instance_id':<38} {'appl':>5} {'resl':>5} {'in_tok':>8} {'out_tok':>8} {'calls':>6} {'time':>7} {'cap':>4}"
        if has_cost:
            hdr += f" {'cost$':>8}"
        print(hdr)
        print("-" * len(hdr))
        for it in result["instances"]:
            line = (
                f"{it['instance_id']:<38} {str(it['patch_applied']):>5} {str(it['resolved']):>5} "
                f"{it.get('input_tokens', 0):>8} {it.get('output_tokens', 0):>8} "
                f"{it.get('llm_calls', 0):>6} {str(it.get('duration_s', 0)) + 's':>7} "
                f"{_fmt_bool(it.get('hit_turn_cap')):>4}"
            )
            if has_cost:
                c = it.get("cost")
                line += f" {('%.4f' % c) if c is not None else '-':>8}"
            print(line)
    else:
        print(f"{'instance_id':<40} {'applied':>8} {'resolved':>9}")
        print("-" * 60)
        for it in result["instances"]:
            print(f"{it['instance_id']:<40} {str(it['patch_applied']):>8} {str(it['resolved']):>9}")

    print("-" * 40)
    print(
        f"resolved {result['resolved']}/{result['total']} "
        f"({result['resolved_rate'] * 100:.1f}%)  |  patch applied "
        f"{result['patch_applied']}/{result['total']}"
    )
    if result.get("empty_patches"):
        print(
            f"  ({result['empty_patches']} empty patch(es) never scored — counted as unresolved; "
            f"rate among the {result['evaluated']} scored: "
            f"{result['resolved_rate_of_evaluated'] * 100:.1f}%)"
        )
    if withp:
        ov, rs, us = result["overall"], result["resolved_stats"], result["unresolved_stats"]
        if "input_tokens" in ov:
            print(
                f"tokens: total in={ov['input_tokens']:,} out={ov['output_tokens']:,}  |  "
                f"avg/instance in={ov['avg_input_tokens']:,} out={ov['avg_output_tokens']:,} "
                f"calls={ov['avg_calls']} time={ov['avg_duration_s']}s"
            )
        if rs.get("n") and "avg_input_tokens" in rs:
            print(
                f"  resolved   (n={rs['n']}): avg in={rs['avg_input_tokens']:,} out={rs['avg_output_tokens']:,} "
                f"calls={rs['avg_calls']} time={rs['avg_duration_s']}s"
            )
        if us.get("n") and "avg_input_tokens" in us:
            print(
                f"  unresolved (n={us['n']}): avg in={us['avg_input_tokens']:,} out={us['avg_output_tokens']:,} "
                f"calls={us['avg_calls']} time={us['avg_duration_s']}s"
            )
        if ov.get("cost") is not None:
            per = round(ov["cost"] / result["resolved"], 4) if result["resolved"] else None
            print(f"cost: total ${ov['cost']}  |  $/resolved {('$' + str(per)) if per is not None else 'n/a'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evals.swebench.summarize_reports")
    p.add_argument("--run-id", help="Run id under ~/logs/run_evaluation/.")
    p.add_argument("--logs-dir", help="Explicit path to a run's logs dir (overrides --run-id).")
    p.add_argument("--predictions", help="predictions.jsonl to join tokens/cost/turn-cap from.")
    p.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens.")
    p.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens.")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Infer hit-turn-cap from _llm_calls when predictions predate the flag.")
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

    predictions = None
    if args.predictions:
        pred_path = Path(os.path.expanduser(args.predictions))
        if not pred_path.exists():
            raise SystemExit(f"predictions file not found: {pred_path}")
        predictions = load_predictions(pred_path)

    result = summarize(
        logs_dir, predictions,
        price_in=args.price_in, price_out=args.price_out, max_turns=args.max_turns,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    _print_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
