"""Tests for the mini eval harness (no LLM calls).

Covers task loading and the execution-based check evaluator. `run_one` (which
drives the real agent) is intentionally not tested here -- it needs a live LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

# evals/ is a top-level dir, not part of the installed package -- make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.runner import load_task, run_checks  # noqa: E402


def test_load_task_parses_fields(tmp_path: Path):
    src = tmp_path / "t.yaml"
    src.write_text(
        "name: demo\n"
        "description: a demo\n"
        "prompt: |\n"
        "  do the thing\n"
        "files:\n"
        "  a.py: |\n"
        "    print(1)\n"
        "checks:\n"
        "  - cmd: python -c \"pass\"\n",
        encoding="utf-8",
    )
    task = load_task(src)
    assert task.name == "demo"
    assert task.prompt.strip() == "do the thing"
    assert task.files["a.py"].strip() == "print(1)"
    assert task.checks == [{"cmd": 'python -c "pass"'}]


def test_load_task_name_defaults_to_filename(tmp_path: Path):
    src = tmp_path / "my_task.yaml"
    src.write_text("prompt: hi\n", encoding="utf-8")
    assert load_task(src).name == "my_task"


def test_check_cmd_exit_code(tmp_path: Path):
    ok, fail = run_checks(tmp_path, [
        {"cmd": "python -c \"import sys; sys.exit(0)\""},
        {"cmd": "python -c \"import sys; sys.exit(1)\""},
    ])
    assert ok.passed is True
    assert fail.passed is False


def test_check_cmd_timeout(tmp_path: Path):
    [res] = run_checks(tmp_path, [
        {"cmd": "python -c \"import time; time.sleep(5)\"", "timeout": 1},
    ])
    assert res.passed is False
    assert "timed out" in res.detail


def test_check_file_contains(tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello world", encoding="utf-8")
    yes, no = run_checks(tmp_path, [
        {"file": "f.txt", "contains": "world"},
        {"file": "f.txt", "contains": "absent"},
    ])
    assert yes.passed is True
    assert no.passed is False


def test_check_file_not_contains_and_exists(tmp_path: Path):
    (tmp_path / "f.txt").write_text("clean", encoding="utf-8")
    nc, exists, missing = run_checks(tmp_path, [
        {"file": "f.txt", "not_contains": "dirty"},
        {"file": "f.txt", "exists": True},
        {"file": "ghost.txt", "exists": False},
    ])
    assert nc.passed is True
    assert exists.passed is True
    assert missing.passed is True  # absent file, exists=False -> pass


def test_check_missing_file_fails(tmp_path: Path):
    [res] = run_checks(tmp_path, [{"file": "nope.txt", "contains": "x"}])
    assert res.passed is False
    assert "missing" in res.detail
