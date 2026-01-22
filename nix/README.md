# ScopePack Installation

## Without Nix (pip)

```bash
# Install package
pip install git+https://github.com/zach-source/context-scope-pack.git

# Or with AST support (recommended for better code chunking)
pip install "scopepack[ast] @ git+https://github.com/zach-source/context-scope-pack.git"

# Download ML models (~500MB, first-time only)
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5')
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
"

# Start daemon manually (or add to shell startup)
scopepack-daemon &

# Copy hooks to your global Claude config
mkdir -p ~/.claude/hooks
cp .claude/hooks/scope_*.py ~/.claude/hooks/
cp .claude/hooks/scope_*.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/scope_*
```

### Running the Daemon

**Manual (foreground):**
```bash
scopepack-daemon --host 127.0.0.1 --port 18765
```

**Background (add to ~/.bashrc or ~/.zshrc):**
```bash
# Start daemon if not running
pgrep -f scopepack-daemon || scopepack-daemon &>/dev/null &
```

**macOS launchd (persistent):**
```bash
cat > ~/Library/LaunchAgents/com.scopepack.daemon.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.scopepack.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/scopepack-daemon</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
</dict>
</plist>
EOF
launchctl load ~/Library/LaunchAgents/com.scopepack.daemon.plist
```

**Linux systemd (persistent):**
```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/scopepack-daemon.service << 'EOF'
[Unit]
Description=ScopePack compression daemon

[Service]
ExecStart=/usr/local/bin/scopepack-daemon
Restart=on-failure

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now scopepack-daemon
```

---

## With Nix Home Manager

Install ScopePack declaratively via Home Manager.

### Quick Start

Add to your `flake.nix` inputs:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager.url = "github:nix-community/home-manager";
    scopepack.url = "github:zach-source/context-scope-pack";
  };
}
```

Import the module in your home configuration:

```nix
{ inputs, ... }:
{
  imports = [
    inputs.scopepack.homeManagerModules.default
  ];

  programs.scopepack = {
    enable = true;
    # All options below are optional with sensible defaults
  };
}
```

## Configuration Options

### Basic

```nix
programs.scopepack = {
  enable = true;  # Enable scopepack

  daemon = {
    enable = true;       # Run daemon as service (default: true)
    port = 18765;        # Daemon port (default: 18765)
    host = "127.0.0.1";  # Daemon host (default: 127.0.0.1)
  };

  hooks = {
    enable = true;           # Install Claude Code hooks (default: true)
    preToolRead = true;      # Compress large file reads (default: true)
    postToolWriteEdit = true; # Track hot files (default: true)
    userPromptSubmit = true;  # Inject context (default: true)
    sessionStart = true;      # Load session state (default: true)
    sessionEnd = true;        # Save session state (default: true)
  };

  settings = {
    maxReadChars = 20000;     # File size threshold for compression
    summaryBudget = 900;      # Target token budget
    cacheDir = "~/.cache/scopepack";  # Cache location
    embedder = "bge-small-en-v1.5";   # Embedding model
  };
};
```

### Embedder Options

```nix
settings.embedder = "bge-small-en-v1.5";  # Local, fast (default)
settings.embedder = "titan-embed-text-v2:0";  # AWS Bedrock Titan
settings.embedder = "cohere-embed-v4:0";  # AWS Bedrock Cohere
```

For Bedrock embedders, ensure `AWS_PROFILE` and `AWS_REGION` are set.

## What Gets Installed

1. **Package**: `scopepack-daemon` and `scopepack-mcp` in PATH
2. **Hooks**: Symlinked to `~/.claude/hooks/scope_*.{py,sh}`
3. **Service**:
   - macOS: launchd agent (`com.scopepack.daemon`)
   - Linux: systemd user service (`scopepack-daemon`)
4. **Environment**: `SCOPE_*` variables in session

## Service Management

### macOS

```bash
# Check status
launchctl list | grep scopepack

# Restart
launchctl kickstart -k gui/$UID/com.scopepack.daemon

# Logs
tail -f ~/.cache/scopepack/daemon.log
```

### Linux

```bash
# Check status
systemctl --user status scopepack-daemon

# Restart
systemctl --user restart scopepack-daemon

# Logs
journalctl --user -u scopepack-daemon -f
```

## First-Time Setup

After enabling, download the ML models:

```bash
# Start a Python shell with scopepack
nix shell github:zach-source/context-scope-pack

# Download models
python -c "
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print('Downloading embedding model...')
SentenceTransformer('BAAI/bge-small-en-v1.5')

print('Downloading summarization model...')
AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

print('Done!')
"
```

## Minimal Example

Just enable with all defaults:

```nix
programs.scopepack.enable = true;
```

This gives you:
- Daemon running on port 18765
- All hooks installed
- ~83% token reduction on large file reads
