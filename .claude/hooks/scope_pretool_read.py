#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""PreToolUse(Read) hook for ScopePack.

When Claude tries to Read a large file:
1. Compute file size / estimated tokens
2. If "too big", generate SCOPE summary (query-aware: current prompt)
3. Return JSON that:
   - sets permissionDecision: "allow"
   - uses updatedInput to read only a small slice (first 200 lines)
   - injects additionalContext with the SCOPE summary
"""

import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

import httpx

# Configuration
MAX_READ_CHARS = int(os.environ.get("SCOPE_MAX_READ_CHARS", "20000"))
SUMMARY_BUDGET_TOKENS = int(os.environ.get("SCOPE_SUMMARY_BUDGET", "900"))
SCOPE_DAEMON_URL = os.environ.get("SCOPE_DAEMON_URL", "http://127.0.0.1:18765")
DAEMON_TIMEOUT = float(os.environ.get("SCOPE_DAEMON_TIMEOUT", "10.0"))

# Paths to skip (secrets, git internals, etc.)
SKIP_PATTERNS = [
    "/.git/",
    ".env",
    "id_rsa",
    "id_ed25519",
    "secrets",
    ".pem",
    ".key",
    "credentials",
    "node_modules/",
    "__pycache__/",
    ".pyc",
]


def file_hash(path: str) -> str:
    """Compute SHA256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_latest_user_prompt(transcript_path: str) -> str:
    """Extract the latest user prompt from the transcript.

    Transcript is JSONL format with messages.
    """
    if not transcript_path:
        return ""

    try:
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return ""

        with open(path, encoding="utf-8") as f:
            # Read last ~500 lines to find recent user messages
            lines = f.readlines()[-500:]

        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Look for user messages
            if obj.get("type") == "user":
                text = obj.get("text", "")
                if isinstance(text, str) and text.strip():
                    return text[:1000]  # Cap query length
    except Exception:
        pass

    return ""


def should_skip_path(path: str) -> bool:
    """Check if path should be skipped (secrets, git, etc.)."""
    return any(pattern in path for pattern in SKIP_PATTERNS)


def compress_via_daemon(
    text: str, query: str, budget_tokens: int, file_path: str
) -> tuple[str | None, str | None]:
    """Call the daemon for SCOPE compression.

    Returns (compressed_text, symbol_index) tuple.
    """
    try:
        with httpx.Client(timeout=DAEMON_TIMEOUT) as client:
            response = client.post(
                f"{SCOPE_DAEMON_URL}/compress",
                json={
                    "text": text,
                    "query": query,
                    "budget_tokens": budget_tokens,
                    "file_path": file_path,
                    "use_cache": True,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("compressed_text"), data.get("symbol_index")
    except Exception:
        pass
    return None, None


def quick_compress(text: str, budget_chars: int) -> str:
    """Fallback compression without daemon (head + tail)."""
    if len(text) <= budget_chars:
        return text

    head_budget = int(budget_chars * 0.7)
    tail_budget = int(budget_chars * 0.2)

    head = text[:head_budget]
    tail = text[-tail_budget:]

    return f"{head}\n\n...[{len(text) - head_budget - tail_budget} chars snipped]...\n\n{tail}"


def main():
    """Main hook entry point."""
    # Read hook event from stdin
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    # Validate this is a PreToolUse for Read
    if data.get("hook_event_name") != "PreToolUse":
        sys.exit(0)
    if data.get("tool_name") != "Read":
        sys.exit(0)

    tool_input = data.get("tool_input", {})
    path = tool_input.get("file_path")

    # Basic validation
    if not path:
        sys.exit(0)
    if not os.path.isabs(path):
        sys.exit(0)

    # Skip sensitive paths
    if should_skip_path(path):
        sys.exit(0)

    # Check if file exists and get size
    try:
        if not os.path.isfile(path):
            sys.exit(0)
        size = os.path.getsize(path)
    except OSError:
        sys.exit(0)

    # If file is small enough, let it through
    if size <= MAX_READ_CHARS:
        sys.exit(0)

    # File is large - generate SCOPE summary
    transcript_path = data.get("transcript_path", "")
    query = get_latest_user_prompt(transcript_path)

    # Read the file
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        sys.exit(0)

    # Try daemon first, fall back to quick compression
    compressed, symbol_index = compress_via_daemon(
        text=text,
        query=query,
        budget_tokens=SUMMARY_BUDGET_TOKENS,
        file_path=path,
    )

    if compressed is None:
        compressed = quick_compress(text, SUMMARY_BUDGET_TOKENS * 4)
        symbol_index = None

    # Build additional context with symbol index for navigation
    symbol_section = ""
    if symbol_index:
        symbol_section = f"\n{symbol_index}\n"

    additional = textwrap.dedent(
        f"""
        [SCOPE SUMMARY: {path}]
        Original size: {len(text):,} chars (~{len(text) // 4:,} tokens)
        Compressed to: ~{len(compressed) // 4:,} tokens

        The file was too large to inject in full. Use the Symbol Index below
        to request specific line ranges (e.g., "show me L44-52").
        {symbol_section}
        --- compressed content ---
        {compressed}
        --- end content ---
        """
    ).strip()

    # Update the tool input to only read first 200 lines
    # This gives Claude exact text to anchor on
    updated_input = dict(tool_input)
    updated_input["offset"] = 0
    updated_input["limit"] = 200

    # Build output
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": (
                f"Large file ({size:,} chars) capped; injected SCOPE summary "
                f"(~{len(compressed) // 4:,} tokens) to reduce context usage."
            ),
            "updatedInput": updated_input,
            "additionalContext": additional,
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
