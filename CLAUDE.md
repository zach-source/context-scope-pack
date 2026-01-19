# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScopePack is a token-efficient context management system for Claude Code. It reduces token usage by compressing large file reads using the SCOPE algorithm (semantic chunking, hybrid relevance scoring, smart truncation, and summarization).

**Key metrics:** 83% token reduction, 91% quality retention, ~600ms latency

## Development Commands

```bash
# Enter development environment (devenv.sh required)
devenv shell

# First-time setup
models-download    # Download ML models (bge-small, distilbart)
db-init           # Initialize SQLite database

# Development
daemon-dev        # Start daemon with auto-reload on localhost:18765
test              # Run pytest (39 tests)
lint              # Format and lint with ruff
typecheck         # Run mypy

# Benchmarks
python -m benchmarks --runners quality --budgets 900
```

Manual setup without devenv:
```bash
pip install -e ".[dev]"
python -m scopepack.daemon --reload
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code Hooks                        │
│  PreToolUse(Read) → PostToolUse(Write/Edit) → Session*      │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP
                    ┌─────────▼─────────┐
                    │   scope-daemon    │
                    │  (localhost:18765)│
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
       ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
       │  Embeddings │ │ Summarizer  │ │   SQLite    │
       │ (bge-small) │ │ (DistilBART)│ │   Cache     │
       └─────────────┘ └─────────────┘ └─────────────┘
```

### Core Modules

- **`src/scopepack/scope.py`** - SCOPE compression algorithm:
  - `chunk_code()` / `chunk_prose()` - Semantic chunking
  - `score_relevance()` - Hybrid BM25 + embedding scoring
  - `_smart_truncate()` - Keyword-aware truncation (preserves query-relevant lines)
  - `compress_with_scope_indexed()` - Full pipeline with symbol index

- **`src/scopepack/embedders.py`** - Multi-provider embedding support:
  - Local: `bge-small-en-v1.5` (default, fast)
  - Bedrock Titan: `titan-embed-text-v2:256/512/1024`
  - Bedrock Cohere: `cohere-embed-v4:256/512/1024/1536`

- **`src/scopepack/daemon.py`** - FastAPI HTTP server, lazy model loading, caching

- **`src/scopepack/mcp_server.py`** - MCP server for Claude Desktop integration

- **`src/scopepack/db.py`** - SQLite async cache for compressed content

### Hooks (`.claude/hooks/`)

- **`scope_pretool_read.py`** - Intercepts large file reads (>20k chars), compresses via daemon
- **`scope_user_prompt_submit.py`** - Injects git status and hot files context
- **`scope_session_start.sh`** - Loads previous session state
- **`scope_session_end.py`** - Saves session state for persistence
- **`scope_posttool_write_edit.py`** - Records file edits for hot file tracking

### Skills (`.claude/skills/`)

- **`compress/`** - Compress large files using SCOPE algorithm
- **`summary/`** - Get quick file summaries

## Key APIs

### Daemon Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, model loading status |
| `/compress` | POST | Full SCOPE compression with caching |
| `/compress/quick` | POST | Fast head+tail truncation (no models) |
| `/summarize` | POST | Standalone summarization |
| `/file-summary` | POST | Get/create cached file summary |

### MCP Server

Start with: `scopepack-mcp` (stdio) or `scopepack-mcp --transport streamable-http --port 8765`

| Tool | Purpose |
|------|---------|
| `compress` | Full SCOPE compression with caching |
| `compress_quick` | Fast head+tail truncation |
| `summarize` | Standalone text summarization |
| `get_file_summary` | Get/create cached file summary |
| `health` | Server status and model info |

### SCOPE Algorithm Flow

1. **Chunk** - `chunk_code()` or `chunk_prose()` based on file type
2. **Score** - `score_relevance()` combines:
   - Semantic: cosine similarity between embeddings
   - Lexical: BM25 keyword matching
   - Parameters: `alpha` (0.5 = balanced), `temperature` (score spread)
3. **Compress** - `compress_chunk()` with `_smart_truncate()`:
   - High-relevance chunks kept mostly intact
   - Low-relevance: smart truncation preserves keyword-matching lines
4. **Index** - `build_symbol_index()` creates navigable line references

### Python Library Usage

```python
from scopepack.scope import compress_with_scope_indexed
from scopepack.embedders import create_embedder

embedder = create_embedder("bge-small-en-v1.5")
compressed, index = compress_with_scope_indexed(
    text=content,
    query="find authentication handlers",
    budget_tokens=900,
    embedder=embedder,
    file_type="code",
    alpha=0.5,        # semantic vs lexical balance
    temperature=2.0,  # score differentiation
)
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCOPE_DAEMON_URL` | `http://127.0.0.1:18765` | Daemon endpoint |
| `SCOPE_CACHE_DIR` | `~/.cache/scopepack` | SQLite and cache location |
| `SCOPE_MAX_READ_CHARS` | `20000` | File size threshold for compression |
| `SCOPE_SUMMARY_BUDGET` | `900` | Target tokens for compressed output |
| `SCOPE_EMBEDDER` | `bge-small-en-v1.5` | Embedding model type |
| `AWS_REGION` | `us-east-1` | For Bedrock embeddings |
| `AWS_PROFILE` | (none) | AWS profile for Bedrock |

## Testing

Tests use pytest. Run all tests:
```bash
pytest tests/test_scope.py -v
```

Run specific test class:
```bash
pytest tests/test_scope.py::TestSmartTruncate -v
pytest tests/test_scope.py::TestHybridScoring -v
```

Test coverage:
- `TestFileTypeDetection` - File type detection
- `TestCodeChunking` - Code semantic chunking
- `TestProseChunking` - Prose/markdown chunking
- `TestCompressionRatios` - Dynamic ratio computation
- `TestQuickCompress` - Fallback compression
- `TestBM25Tokenization` - Tokenizer for BM25
- `TestBM25Scoring` - Lexical relevance scoring
- `TestHybridScoring` - Combined semantic + lexical scoring
- `TestSmartTruncate` - Keyword-aware truncation
- `TestCompressChunk` - Chunk compression with query
