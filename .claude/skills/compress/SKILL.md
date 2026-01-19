---
name: ScopePack Compress
description: This skill should be used when the user asks to "compress a file", "reduce tokens", "summarize large file for context", "scope compress", "make file smaller for Claude", or wants to reduce token usage for large files before adding them to context.
version: 0.1.0
---

# ScopePack Compress Skill

Compress large files using the SCOPE algorithm to reduce token usage while preserving important content.

## Overview

ScopePack uses semantic chunking, relevance scoring via embeddings, and dynamic compression to achieve:
- **83% token reduction** (6x compression)
- **91% quality retention** (vs 30% with naive truncation)

## When to Use

Use this skill when:
- A file is too large to fit in context (>20KB)
- Token budget is limited and content must be compressed
- Need to preserve query-relevant portions of code/docs

## Prerequisites

Ensure the ScopePack daemon is running:

```bash
# Start daemon (in scopepack directory)
cd /path/to/scopepack
devenv shell -- daemon-dev

# Or without devenv
python -m scopepack.daemon
```

The daemon runs on `http://127.0.0.1:18765` by default.

## Compression Workflow

### Step 1: Check File Size

Determine if compression is needed:

```python
import os
file_size = os.path.getsize(file_path)
if file_size < 20000:  # 20KB threshold
    # File is small enough, read directly
else:
    # Compress before reading
```

### Step 2: Compress via HTTP API

Call the ScopePack daemon:

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:18765/compress",
    json={
        "text": file_content,
        "query": "what you're looking for",  # Improves relevance
        "budget_tokens": 900,
        "file_path": file_path,
        "use_cache": True,
    },
    timeout=15.0,
)

data = response.json()
compressed = data["compressed_text"]
symbol_index = data.get("symbol_index", "")
```

### Step 3: Use Symbol Index for Navigation

The symbol index provides line-number references:

```
## Symbol Index
- L1-25: class MyClass [highly relevant]
- L27-45: def process_data [relevant]
- L47-80: def helper_function [low relevance]
```

To read specific sections, use line ranges:

```python
# Read lines 27-45 for process_data function
Read(file_path, offset=27, limit=19)
```

## Quick Compression (No ML Models)

For faster compression without loading ML models:

```python
response = httpx.post(
    "http://127.0.0.1:18765/compress/quick",
    json={
        "text": file_content,
        "budget_tokens": 900,
        "file_path": file_path,
    },
)
```

Quick compression uses head+tail truncation. Faster but lower quality.

## MCP Integration

If using ScopePack MCP server instead of HTTP daemon:

```bash
# Start MCP server
scopepack-mcp --transport stdio
```

The MCP server exposes the same tools:
- `compress` - Full SCOPE compression
- `compress_quick` - Fast head+tail truncation
- `summarize` - Standalone summarization
- `get_file_summary` - Cached file summaries
- `health` - Server status

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCOPE_DAEMON_URL` | `http://127.0.0.1:18765` | Daemon endpoint |
| `SCOPE_SUMMARY_BUDGET` | `900` | Target tokens |
| `SCOPE_EMBEDDER` | `bge-small-en-v1.5` | Embedding model |

## Tips for Best Results

1. **Provide a specific query** - The query parameter significantly improves relevance scoring
2. **Use caching** - Repeated compressions of the same content are instant
3. **Check symbol index** - Navigate to specific functions/classes when needed
4. **Adjust budget** - Lower budget = more compression, higher budget = more content
