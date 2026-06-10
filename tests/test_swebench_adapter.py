"""Offline tests for the SWE-bench adapter (no network / LLM / datasets).

Covers the pure plumbing: prompt building and git-diff patch extraction on a
local throwaway repo. Skipped entirely if git isn't on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.swebench.run_swebench import build_prompt, extract_patch  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def _init_repo(path: Path):
    _git(["init", "-q"], path)
    _git(["config", "user.email", "t@t.t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)


def test_build_prompt_includes_issue_and_repo():
    inst = {"repo": "psf/requests", "problem_statement": "  Something is broken.  "}
    prompt = build_prompt(inst)
    assert "psf/requests" in prompt
    assert "Something is broken." in prompt
    # The agent must be told not to touch tests.
    assert "test" in prompt.lower()


def test_extract_patch_captures_edits(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    # Edit an existing file + add a new one.
    (repo / "mod.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    (repo / "new.py").write_text("x = 1\n", encoding="utf-8")

    patch = extract_patch(repo)
    assert "diff --git" in patch
    assert "-    return 1" in patch
    assert "+    return 2" in patch
    assert "new.py" in patch  # staged, so new files are captured


def test_extract_patch_empty_when_clean(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    assert extract_patch(repo).strip() == ""
