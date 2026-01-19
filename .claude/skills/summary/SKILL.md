---
name: ScopePack Summary
description: This skill should be used when the user asks to "summarize a file", "get file summary", "what does this file do", "overview of file", or needs a quick understanding of a file's contents without reading the full content.
version: 0.1.0
---

# ScopePack Summary Skill

Generate concise summaries of files using ScopePack's summarization capabilities.

## Overview

ScopePack provides file summarization that:
- Uses DistilBART model for intelligent summarization
- Caches summaries for instant retrieval
- Falls back to smart truncation when models unavailable

## When to Use

Use this skill when:
- Need quick overview of a file's purpose
- Exploring a codebase to understand structure
- Building context without reading full files
- Creating documentation or onboarding materials

## Prerequisites

Ensure the ScopePack daemon is running:

```bash
# Start daemon
cd /path/to/scopepack
devenv shell -- daemon-dev
```

## Get File Summary

### Via HTTP API

```python
import httpx
import hashlib

# Read file and compute hash
with open(file_path, "r") as f:
    content = f.read()

file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

response = httpx.post(
    "http://127.0.0.1:18765/file-summary",
    json={
        "file_path": file_path,
        "file_hash": file_hash,
        "content": content,  # Optional if daemon can read file
    },
    timeout=15.0,
)

data = response.json()
summary = data["summary"]
cache_hit = data["cache_hit"]
```

### Via MCP Tool

If using the ScopePack MCP server:

```
Tool: get_file_summary
Arguments:
  file_path: "/path/to/file.py"
  content: ""  # Optional, will read file if empty
```

## Standalone Summarization

For summarizing arbitrary text (not files):

```python
response = httpx.post(
    "http://127.0.0.1:18765/summarize",
    json={
        "text": long_text_content,
        "max_tokens": 150,
    },
)

summary = response.json()["summary"]
```

## Caching Behavior

File summaries are cached by:
- File path
- File content hash
- Model version

Benefits:
- Instant retrieval for unchanged files
- Automatic invalidation when content changes
- Persists across sessions via SQLite

## Summary Quality

The summarization model (DistilBART) produces:
- Concise summaries (typically 50-150 tokens)
- Coherent prose (not bullet points)
- Focus on main purpose/functionality

For code files, summaries typically describe:
- Primary purpose of the file
- Key functions/classes
- External dependencies
- Usage patterns

## Fallback Behavior

When summarization model is unavailable:
- Falls back to quick_compress (head+tail truncation)
- Lower quality but still usable
- No ML model loading required

## Integration with Compression

Combine summaries with compression for context building:

1. **Get summary** - Quick understanding of file purpose
2. **Compress if needed** - Add to context with token budget
3. **Use symbol index** - Navigate to specific sections

```python
# Get summary first
summary_response = httpx.post(
    "http://127.0.0.1:18765/file-summary",
    json={"file_path": file_path, "file_hash": file_hash},
)

# If relevant, compress for full context
if is_relevant(summary_response.json()["summary"]):
    compress_response = httpx.post(
        "http://127.0.0.1:18765/compress",
        json={
            "text": content,
            "query": "specific query",
            "budget_tokens": 900,
        },
    )
```

## Tips

1. **Cache warmup** - Summarize frequently-used files at session start
2. **Hash consistency** - Use same hashing algorithm for cache hits
3. **Budget summaries** - Summaries are typically 50-150 tokens
4. **Combine with grep** - Summary helps decide if file is relevant before deeper search
