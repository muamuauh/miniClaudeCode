# SWE-bench Lite adapter

Run miniClaudeCode against [SWE-bench Lite](https://www.swebench.com/) — 300 real
GitHub issues from popular Python projects. The agent must produce a code patch
that makes the hidden tests pass.

The workflow is **two stages** (the standard SWE-bench split):

```
   stage 1: generate predictions          stage 2: score them
   (this adapter, runs natively)          (official harness, runs in Docker)
   ┌───────────────────────────┐          ┌──────────────────────────────┐
   │ clone repo @ base_commit  │          │ for each prediction:         │
   │ run agent on the issue    │  ──────► │  apply model patch           │
   │ git diff = model_patch    │ predictions  apply gold test patch      │
   │ write predictions.jsonl   │  .jsonl  │  run FAIL_TO_PASS / PASS_..  │
   └───────────────────────────┘          │  resolved = all pass         │
                                          └──────────────────────────────┘
```

Why split? Each repo needs a specific Python version + pinned dependencies to run
its tests. The official harness handles that hell with per-instance Docker images.
Our agent only needs to *read and edit* the code, so it runs natively — no env
setup required to produce a patch.

## Prerequisites

```powershell
conda activate miniClaudeCode
pip install datasets          # to load the dataset (stage 1)
pip install swebench          # to score (stage 2) — also needs Docker running
```

`git` and Docker Desktop must be installed. Stage 1 fetches repos by commit SHA
from GitHub, so it needs network access.

## Stage 1 — generate predictions

```powershell
$env:PYTHONUTF8="1"

# Sanity-check the whole pipeline with NO LLM (emits each instance's gold patch).
# These should later score as RESOLVED — proves clone/diff/format/Docker all work.
python -m evals.swebench.run_swebench --gold --limit 5

# Real run: agent solves N instances (costs tokens + time; thinking models are slow).
python -m evals.swebench.run_swebench --limit 1
python -m evals.swebench.run_swebench --instance marshmallow-code__marshmallow-1359 --verbose
python -m evals.swebench.run_swebench --limit 10 --profile deepseek --max-turns 40
```

Output: `evals/swebench/out/predictions.jsonl`, one line per instance:

```json
{"instance_id": "...", "model_name_or_path": "miniclaudecode", "model_patch": "diff --git ..."}
```

(Extra `_`-prefixed fields — tokens, calls, errors — are diagnostics the official
harness ignores.)

### Useful flags

| Flag | Meaning |
|---|---|
| `--limit N` | run the first N instances (default 1) |
| `--instance ID` | run a specific instance (repeatable) |
| `--gold` | emit gold patches instead of running the agent (no LLM) |
| `--profile` / `--model` | pick the LLM (same as the CLI) |
| `--max-turns N` | per-instance agent turn cap (default 40) |
| `--keep-clones` | keep temp clones for debugging |
| `--verbose` | stream live agent output |

## Stage 2 — score the predictions

The grader computes the **resolved rate** (the headline SWE-bench metric): it
applies each model patch + the gold test patch in a per-repo Docker image and
runs the target tests.

> ⚠️ **Windows note.** The official `swebench` harness does **not run natively on
> Windows** — it does `import resource` (a Unix-only stdlib module) at import
> time, so `python -m swebench.harness.run_evaluation` fails with
> `ModuleNotFoundError: No module named 'resource'` before Docker is ever
> touched. Use one of the two paths below.

### Option A — WSL2 (local Docker scoring)

Docker Desktop shares its daemon with WSL2, so run the harness from inside an
Ubuntu WSL2 shell:

```bash
# inside WSL2 (Ubuntu)
pip install swebench
cd /mnt/e/codes/miniClaudeCode
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path evals/swebench/out/predictions.jsonl \
  --max_workers 4 \
  --run_id mcc-run
```

First run builds/pulls a Docker image per repo (large, slow). It writes a report
JSON with the resolved rate and per-instance results.

### Option B — sb-cli (cloud scoring, no local Docker)

The SWE-bench team hosts a cloud evaluator — easiest on Windows:

```powershell
pip install sb-cli
# get a free API key: https://www.swebench.com/sb-cli/
sb-cli submit swe-bench_lite test --predictions_path evals/swebench/out/predictions.jsonl
```

It runs the same evaluation server-side and returns the resolved rate.

### Sanity-gate with --gold first

Before spending tokens on real runs, score the `--gold` predictions — they
should come back ~100% resolved. If they don't, your scoring setup (not the
agent) is the problem.

## Cost & scope notes

- Real runs cost tokens and are **slow** (cloning big repos + many agent turns;
  thinking models like qwen are slower still). Start with `--limit 1`.
- This adapter generates the **model patch only**; it never touches the gold test
  patch — the harness applies that itself, so the agent can't game the tests.
- The prompt tells the agent not to edit test files; if it does anyway, those
  edits are still captured in the diff but the harness resets tests to gold.
- For a quick offline pipeline check (no network/LLM), see
  `tests/test_swebench_adapter.py`.
