#!/usr/bin/env bash
# SessionStart hook for ScopePack
#
# Loads previous session state pack if available.
# Also ensures the daemon is running.

set -e

SCOPE_DAEMON_URL="${SCOPE_DAEMON_URL:-http://127.0.0.1:18765}"

# Check if daemon is running
check_daemon() {
    curl -s --max-time 2 "${SCOPE_DAEMON_URL}/health" > /dev/null 2>&1
}

# Try to start daemon if not running
start_daemon() {
    if command -v python3 &> /dev/null; then
        # Start in background, redirect output
        nohup python3 -m scopepack.daemon > /tmp/scopepack-daemon.log 2>&1 &
        sleep 2
    fi
}

# Main
main() {
    # Read event from stdin (required but we don't use it much here)
    EVENT=$(cat)

    # Extract cwd from event
    CWD=$(echo "$EVENT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('cwd', ''))" 2>/dev/null || echo "")

    # Ensure daemon is running
    if ! check_daemon; then
        start_daemon
    fi

    # Try to load previous session state
    if check_daemon && [ -n "$CWD" ]; then
        RESPONSE=$(curl -s --max-time 5 \
            -X GET \
            "${SCOPE_DAEMON_URL}/session-state?project_dir=${CWD}" \
            2>/dev/null || echo "{}")

        STATE_PACK=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('state_pack', ''))" 2>/dev/null || echo "")

        if [ -n "$STATE_PACK" ] && [ "$STATE_PACK" != "null" ]; then
            # Inject previous session context
            cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "[Previous Session State]\n${STATE_PACK}\n[/Previous Session State]"
  }
}
EOF
            exit 0
        fi
    fi

    # No state to inject
    exit 0
}

main
