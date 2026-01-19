#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""SessionEnd hook for ScopePack.

Saves a compressed session state pack for the next session.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

import httpx

SCOPE_DAEMON_URL = os.environ.get("SCOPE_DAEMON_URL", "http://127.0.0.1:18765")
DAEMON_TIMEOUT = float(os.environ.get("SCOPE_DAEMON_TIMEOUT", "5.0"))


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


def build_state_pack(cwd: str, session_id: str) -> dict:
    """Build a state pack to save for next session."""
    # Git info
    branch = run_git_cmd(["git", "branch", "--show-current"], cwd)
    commit = run_git_cmd(["git", "rev-parse", "--short", "HEAD"], cwd)
    status = run_git_cmd(["git", "status", "--porcelain"], cwd)

    # Parse changed files
    changed_files = []
    for line in status.split("\n"):
        if line.strip():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                changed_files.append(parts[1])

    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "git_branch": branch,
        "git_commit": commit,
        "uncommitted_files": changed_files[:20],
        "project_dir": cwd,
    }


def save_state_pack(state: dict, cwd: str) -> None:
    """Save state pack to daemon."""
    try:
        with httpx.Client(timeout=DAEMON_TIMEOUT) as client:
            client.post(
                f"{SCOPE_DAEMON_URL}/session-state",
                json={
                    "session_id": state["session_id"],
                    "project_dir": cwd,
                    "state_pack": state,
                    "git_branch": state.get("git_branch"),
                    "git_commit": state.get("git_commit"),
                },
            )
    except Exception:
        pass  # Non-critical


def main():
    """Main hook entry point."""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    if data.get("hook_event_name") != "SessionEnd":
        sys.exit(0)

    cwd = data.get("cwd", os.getcwd())
    session_id = data.get("session_id", "unknown")

    # Build and save state pack
    state = build_state_pack(cwd, session_id)
    save_state_pack(state, cwd)

    # No output needed
    sys.exit(0)


if __name__ == "__main__":
    main()
