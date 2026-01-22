# ScopePack

Token-efficient context management for Claude Code via SCOPE-style compression.

**Key Results:**
- 🗜️ **83% token reduction** (6x compression)
- 📊 **91% quality retention** (vs 30% with naive truncation)
- ⚡ **~600ms latency** per compression

## Overview

ScopePack reduces token burn in Claude Code sessions by:

1. **Semantic compression** - SCOPE algorithm chunks code into functions/classes, scores relevance via embeddings, compresses low-relevance chunks more aggressively
2. **Smart truncation** - Preserves query-relevant lines even in the middle of functions
3. **Symbol indexing** - Generates navigable line-number references for compressed content
4. **Caching** - SQLite cache avoids recomputing for identical content

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/zach-source/context-scope-pack
cd context-scope-pack

# Install with pip
pip install -e ".[all]"

# Or using devenv (recommended for development)
devenv shell
models-download  # First-time: download ML models (~500MB)
db-init          # Initialize SQLite cache
```

### Nix / Home Manager (Recommended)

Declarative installation with automatic daemon service:

```nix
# flake.nix
{
  inputs.scopepack.url = "github:zach-source/context-scope-pack";
}

# home.nix
{
  imports = [ inputs.scopepack.homeManagerModules.default ];
  programs.scopepack.enable = true;
}
```

This installs hooks to `~/.claude/hooks/` and runs the daemon via launchd (macOS) or systemd (Linux).

See [nix/README.md](nix/README.md) for full configuration options.

### Choose Your Integration

| Use Case | Integration | Setup Time |
|----------|-------------|------------|
| Nix/Home Manager | [Nix Module](#nix--home-manager-recommended) | 1 min |
| Claude Code CLI | [Hooks](#claude-code-cli-hooks) | 2 min |
| Claude Desktop | [MCP Server](#mcp-server-claude-desktop) | 1 min |
| Custom Agents (SDK) | [HTTP API](#claude-sdk-http-api) or [Python Library](#claude-sdk-python-library) | 5 min |

---

## Claude Code CLI (Hooks)

Automatically compress large files when Claude Code reads them.

### Setup

**Step 1:** Copy hooks to your project

```bash
cp -r scopepack/.claude/hooks /path/to/your/project/.claude/
cp -r scopepack/.claude/skills /path/to/your/project/.claude/  # Optional
```

**Step 2:** Configure `.claude/settings.json`

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Read",
      "hooks": [{
        "type": "command",
        "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/scope_pretool_read.py"
      }]
    }]
  }
}
```

**Step 3:** Start the daemon (in a separate terminal)

```bash
# Using devenv
cd scopepack && devenv shell -- daemon-dev

# Or directly
scopepack-daemon
```

### How It Works

When Claude tries to read a large file (>20KB):

```
┌─────────────────────────────────────────────────────────────────┐
│  Claude Code reads large_file.py                                │
│       ↓                                                         │
│  PreToolUse hook intercepts → Daemon compresses                 │
│       ↓                                                         │
│  Claude sees compressed summary + symbol index                  │
│       ↓                                                         │
│  Claude can request specific line ranges if needed              │
└─────────────────────────────────────────────────────────────────┘
```

### Skills (Optional)

Copy skills for manual compression commands:

```bash
cp -r scopepack/.claude/skills /path/to/your/project/.claude/
```

Available skills:
- **compress** - "compress this file", "reduce tokens", "scope compress"
- **summary** - "summarize this file", "what does this file do"

---

## MCP Server (Claude Desktop)

Use ScopePack as an MCP server with Claude Desktop or any MCP client.

### Setup

**Step 1:** Install ScopePack

```bash
pip install scopepack
# Or: pip install -e ".[all]" from source
```

**Step 2:** Add to Claude Desktop config

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "scopepack": {
      "command": "scopepack-mcp",
      "args": []
    }
  }
}
```

**Step 3:** Restart Claude Desktop

The following tools become available:

| Tool | Description |
|------|-------------|
| `compress` | Full SCOPE compression with caching |
| `compress_quick` | Fast head+tail truncation (no ML models) |
| `compress_mcp_output` | Compress output from other MCP tools (Context7, GitHub, etc.) |
| `summarize` | Standalone text summarization |
| `get_file_summary` | Get/create cached file summary |
| `health` | Server status and model info |

### Compressing MCP Output (e.g., Context7)

Use `compress_mcp_output` to compress large results from other MCP tools:

```
1. Call mcp__context7__get-library-docs → returns 5000 tokens of docs
2. Call mcp__scopepack__compress_mcp_output with:
   - content: the docs from step 1
   - query: "React hooks useState" (your original question)
   - source: "context7"
3. Get compressed docs (~1500 tokens) with relevance preserved
```

### HTTP Transport (Alternative)

For web clients or custom integrations:

```bash
scopepack-mcp --transport streamable-http --port 8765
```

---

## Claude SDK (HTTP API)

For custom agents built with the [Claude Agent SDK](https://github.com/anthropics/anthropic-sdk-python).

### Setup

**Step 1:** Start the daemon

```bash
# Background process
scopepack-daemon &

# Or with auto-reload for development
cd scopepack && devenv shell -- daemon-dev
```

**Step 2:** Call from your agent

```python
import httpx
from anthropic import Anthropic

client = Anthropic()
SCOPE_URL = "http://127.0.0.1:18765"

def compress_file_for_context(file_path: str, query: str) -> str:
    """Compress a large file before adding to agent context."""
    with open(file_path) as f:
        content = f.read()

    # Skip small files
    if len(content) < 20000:
        return content

    # Call ScopePack daemon
    response = httpx.post(
        f"{SCOPE_URL}/compress",
        json={
            "text": content,
            "query": query,  # What the agent is looking for
            "budget_tokens": 900,
            "file_path": file_path,
            "use_cache": True,
        },
        timeout=15.0,
    )
    data = response.json()

    # Format for agent context
    symbol_index = data.get("symbol_index", "")
    compressed = data["compressed_text"]

    return f"""[Compressed from {data['original_tokens']} to {data['compressed_tokens']} tokens]
{symbol_index}

{compressed}"""


# Example: Use in agent loop
def run_agent(user_query: str, files: list[str]):
    # Compress files for context
    context_parts = []
    for file_path in files:
        compressed = compress_file_for_context(file_path, user_query)
        context_parts.append(f"## {file_path}\n{compressed}")

    context = "\n\n".join(context_parts)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"{user_query}\n\nContext:\n{context}"
        }]
    )
    return response.content[0].text
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, model status |
| `/compress` | POST | Full SCOPE compression |
| `/compress/quick` | POST | Fast head+tail truncation |
| `/summarize` | POST | Standalone summarization |
| `/file-summary` | POST | Get/create cached file summary |

### Request/Response Example

```bash
curl -X POST http://127.0.0.1:18765/compress \
  -H "Content-Type: application/json" \
  -d '{
    "text": "def hello(): print(\"hello\")\ndef world(): print(\"world\")",
    "query": "greeting function",
    "budget_tokens": 900,
    "file_path": "hello.py",
    "use_cache": true
  }'
```

Response:
```json
{
  "compressed_text": "def hello(): print(\"hello\")...",
  "symbol_index": "## Symbol Index\n- L1: def hello [highly relevant]\n- L2: def world [low relevance]",
  "original_tokens": 50,
  "compressed_tokens": 25,
  "compression_ratio": 0.5,
  "cache_hit": false,
  "model_version": "v2-bge-small-en-v1.5"
}
```

---

## Claude SDK (Python Library)

Use SCOPE directly without a daemon (embedded in your agent).

### Basic Usage

```python
from scopepack.scope import compress_with_scope_indexed, detect_file_type
from scopepack.embedders import create_embedder

# Create embedder once (reuse across calls)
embedder = create_embedder("bge-small-en-v1.5")

def compress_for_agent(content: str, query: str, file_path: str = "") -> str:
    """Compress content using SCOPE algorithm."""
    file_type = detect_file_type(file_path) if file_path else "code"

    compressed, symbol_index = compress_with_scope_indexed(
        text=content,
        query=query,
        budget_tokens=900,
        embedder=embedder,
        summarizer=None,  # Uses smart truncation
        file_type=file_type,
        alpha=0.5,        # Balance: 1.0=semantic, 0.0=lexical
        temperature=2.0,  # Score differentiation
    )

    return f"{symbol_index}\n\n{compressed}"
```

### Quick Compression (No ML Models)

For fast compression without loading ML models:

```python
from scopepack.scope import quick_compress_indexed, detect_file_type

def quick_compress(content: str, file_path: str = "") -> str:
    """Fast compression using head+tail truncation."""
    file_type = detect_file_type(file_path) if file_path else "code"
    compressed, symbol_index = quick_compress_indexed(
        content,
        budget_chars=3600,  # ~900 tokens
        file_type=file_type,
    )
    return f"{symbol_index}\n\n{compressed}"
```

### Compressing MCP Output in Agents

Use the `mcp_utils` module to compress MCP tool results in your agent loop:

```python
from scopepack.mcp_utils import (
    compress_mcp_result,
    compress_context7_result,
    MCPResultCompressor,
)

# Option 1: Direct function call
def handle_context7_result(docs: str, user_query: str) -> str:
    """Compress Context7 docs before adding to context."""
    return compress_context7_result(
        content=docs,
        topic=user_query,
        budget_tokens=1500,
    )

# Option 2: Stateful compressor for agent loops
compressor = MCPResultCompressor(budget_tokens=1500)

for tool_result in mcp_results:
    if tool_result["tool"].startswith("mcp__context7"):
        content = compressor.compress(
            tool_result["content"],
            query=user_query,
            source="context7",
        )
```

### AWS Bedrock Embeddings

Use cloud embeddings instead of local models:

```python
from scopepack.embedders import create_embedder, EmbedderType
from scopepack.scope import compress_with_scope_indexed

# Titan embeddings (fast, cheap)
embedder = create_embedder(
    EmbedderType.TITAN_V2_256,  # 256/512/1024 dimensions
    region="us-east-1",
    profile="my-aws-profile",  # Optional
)

# Cohere v4 (highest quality)
embedder = create_embedder(
    EmbedderType.COHERE_V4_1024,
    region="us-west-2",
)

# Use with compress
compressed, index = compress_with_scope_indexed(
    text=content,
    query="authentication handlers",
    budget_tokens=900,
    embedder=embedder,
    file_type="code",
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Integration Options                          │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ CLI Hooks    │ MCP Server   │ HTTP API     │ Python Library    │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────────┘
       │              │              │               │
       └──────────────┴──────────────┴───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   SCOPE Engine    │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
       ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
       │  Embeddings │ │ Summarizer  │ │   SQLite    │
       │ (bge-small) │ │ (BART)      │ │   Cache     │
       └─────────────┘ └─────────────┘ └─────────────┘
```

## SCOPE Algorithm

1. **Semantic Chunking** - Split into functions, classes, paragraphs
2. **Hybrid Scoring** - BM25 lexical + embedding semantic similarity
3. **Smart Truncation** - Preserve query-relevant lines (not just head/tail)
4. **Dynamic Compression** - High relevance → less compression
5. **Symbol Index** - Generate navigable line references

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCOPE_DAEMON_URL` | `http://127.0.0.1:18765` | Daemon endpoint |
| `SCOPE_DAEMON_PORT` | `18765` | Daemon port |
| `SCOPE_CACHE_DIR` | `~/.cache/scopepack` | Cache directory |
| `SCOPE_MAX_READ_CHARS` | `20000` | Compress files larger than this |
| `SCOPE_SUMMARY_BUDGET` | `900` | Target tokens for summaries |
| `SCOPE_EMBEDDER` | `bge-small-en-v1.5` | Embedding model |
| `AWS_REGION` | `us-east-1` | For Bedrock embeddings |
| `AWS_PROFILE` | (none) | AWS profile for Bedrock |

---

## Benchmarks

```bash
# Run quality benchmarks
devenv shell -- python -m benchmarks --runners quality --budgets 900

# Compare embedding models (requires AWS credentials)
assume my-profile --exec "python -m benchmarks --runners quality"
```

### Results (budget=900 tokens)

| Metric | SCOPE | Quick (naive) |
|--------|-------|---------------|
| Avg Quality | 91.4% | 30.5% |
| Win Rate | 90.9% | 0% |
| Questions Answerable | 9/11 | 1/11 |

---

## Development

```bash
# Enter dev environment
devenv shell

# Available commands
daemon-dev      # Start HTTP daemon with auto-reload
mcp-server      # Start MCP server (stdio)
test            # Run pytest (39 tests)
lint            # Format and lint with ruff
typecheck       # Run mypy
db-init         # Initialize SQLite database
models-download # Download ML models (~500MB)
```

---

## License

MIT
