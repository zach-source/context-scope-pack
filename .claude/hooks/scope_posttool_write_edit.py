#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""PostToolUse(Write|Edit) hook for ScopePack.

When Claude writes or edits a file:
1. Invalidate cached summaries for that file
2. Track file as "hot" (frequently edited)
3. Optionally regenerate summary in background
"""

import json
import os
import sys

import httpx

SCOPE_DAEMON_URL = os.environ.get("SCOPE_DAEMON_URL", "http://127.0.0.1:18765")
DAEMON_TIMEOUT = float(os.environ.get("SCOPE_DAEMON_TIMEOUT", "5.0"))


def notify_daemon_file_changed(file_path: str, project_dir: str) -> None:
    """Notify daemon that a file was modified."""
    try:
        with httpx.Client(timeout=DAEMON_TIMEOUT) as client:
            client.post(
                f"{SCOPE_DAEMON_URL}/file-changed",
                json={"file_path": file_path, "project_dir": project_dir},
            )
    except Exception:
        pass  # Non-critical


def main():
    """Main hook entry point."""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    if data.get("hook_event_name") != "PostToolUse":
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    if tool_name not in ("Write", "Edit"):
        sys.exit(0)

    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path")

    if not file_path:
        sys.exit(0)

    # Get project directory from cwd
    cwd = data.get("cwd", os.getcwd())

    # Notify daemon (fire and forget)
    notify_daemon_file_changed(file_path, cwd)

    # No output needed - just let the tool result pass through
    sys.exit(0)


if __name__ == "__main__":
    main()
