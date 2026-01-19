#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""UserPromptSubmit hook for ScopePack.

Injects a query-aware ScopePack into context:
- Git status/diff summary
- Hot files (recently edited)
- Cached file summaries
- Replay buffer facts (if available)
"""

import json
import os
import subprocess
import sys

import httpx

SCOPE_DAEMON_URL = os.environ.get("SCOPE_DAEMON_URL", "http://127.0.0.1:18765")
DAEMON_TIMEOUT = float(os.environ.get("SCOPE_DAEMON_TIMEOUT", "5.0"))
SCOPEPACK_BUDGET_TOKENS = int(os.environ.get("SCOPE_PACK_BUDGET", "500"))


def run_git_cmd(cmd: list[str], cwd: str) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_git_status(cwd: str) -> dict:
    """Get git repository status."""
    # Check if in a git repo
    if not run_git_cmd(["git", "rev-parse", "--git-dir"], cwd):
        return {}

    branch = run_git_cmd(["git", "branch", "--show-current"], cwd)
    status = run_git_cmd(["git", "status", "--porcelain"], cwd)
    diff_stat = run_git_cmd(["git", "diff", "--stat", "--no-color"], cwd)

    # Parse changed files
    changed_files = []
    for line in status.split("\n"):
        if line.strip():
            # Format: XY filename
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                changed_files.append({"status": parts[0], "file": parts[1]})

    return {
        "branch": branch,
        "is_dirty": bool(status),
        "changed_files": changed_files[:10],  # Cap at 10
        "diff_stat": diff_stat[:500] if diff_stat else None,
    }


def get_hot_files(cwd: str) -> list[dict]:
    """Get hot files from daemon cache."""
    try:
        with httpx.Client(timeout=DAEMON_TIMEOUT) as client:
            response = client.get(
                f"{SCOPE_DAEMON_URL}/hot-files",
                params={"project_dir": cwd, "limit": 5},
            )
            if response.status_code == 200:
                return response.json().get("files", [])
    except Exception:
        pass
    return []


def compress_diff(diff: str, budget: int) -> str:
    """Compress git diff to fit budget."""
    if not diff or len(diff) <= budget:
        return diff

    # Keep first and last parts
    head = diff[: budget // 2]
    tail = diff[-(budget // 3) :]
    return f"{head}\n...[diff truncated]...\n{tail}"


def build_scopepack(cwd: str, _prompt: str) -> str:
    """Build a ScopePack for the current context."""
    git_info = get_git_status(cwd)
    hot_files = get_hot_files(cwd)

    parts = ["[SCOPEPACK v1]"]

    # Git status
    if git_info:
        branch = git_info.get("branch", "unknown")
        dirty = "dirty" if git_info.get("is_dirty") else "clean"
        parts.append(f"Repo: {branch} ({dirty})")

        if git_info.get("changed_files"):
            parts.append("Changed files:")
            for f in git_info["changed_files"][:5]:
                parts.append(f"  {f['status']} {f['file']}")

        if git_info.get("diff_stat"):
            parts.append("Diff summary:")
            parts.append(compress_diff(git_info["diff_stat"], 300))

    # Hot files
    if hot_files:
        parts.append("Hot files (recently accessed):")
        for f in hot_files[:5]:
            parts.append(f"  - {f.get('file_path', 'unknown')}")

    parts.append("[/SCOPEPACK]")

    return "\n".join(parts)


def main():
    """Main hook entry point."""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    if data.get("hook_event_name") != "UserPromptSubmit":
        sys.exit(0)

    prompt = data.get("prompt", "")
    cwd = data.get("cwd", os.getcwd())

    # Build ScopePack
    scopepack = build_scopepack(cwd, prompt)

    # Only inject if we have meaningful content
    if len(scopepack) > 50:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": scopepack,
            }
        }
        print(json.dumps(output))
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
