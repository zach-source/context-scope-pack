"""ScopePack MCP Server - Model Context Protocol integration.

Exposes SCOPE compression capabilities via MCP tools for Claude Desktop
and other MCP clients.

Usage:
    # Run directly
    python -m scopepack.mcp_server

    # Or via entry point
    scopepack-mcp

    # With stdio transport (default)
    scopepack-mcp --transport stdio

    # With HTTP transport
    scopepack-mcp --transport streamable-http --port 8765
"""

import logging
import os
import sys
from typing import Annotated

from mcp.server.fastmcp import FastMCP

from .db import CacheDB, CacheEntry, FileSummary, content_hash, init_db
from .scope import (
    compress_with_scope_indexed,
    detect_file_type,
    quick_compress_indexed,
)

# Configuration
EMBEDDER_TYPE = os.environ.get("SCOPE_EMBEDDER", "bge-small-en-v1.5")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE")
MODEL_VERSION = f"v2-{EMBEDDER_TYPE}"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scopepack.mcp")

# Initialize FastMCP server
mcp = FastMCP("ScopePack")

# Database and models (lazy-loaded)
db = CacheDB()
_embedder = None
_summarizer = None


async def _ensure_db() -> None:
    """Ensure database is initialized."""
    await init_db()


async def _get_embedder():
    """Get or load the embedding model."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDER_TYPE}...")
        from .embedders import create_embedder

        _embedder = create_embedder(
            EMBEDDER_TYPE,
            region=AWS_REGION,
            profile=AWS_PROFILE,
        )
        logger.info(f"Embedding model loaded: {EMBEDDER_TYPE}")
    return _embedder


async def _get_summarizer():
    """Get or load the summarization model."""
    global _summarizer
    if _summarizer is None:
        try:
            logger.info("Loading summarization model...")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            _summarizer = (model, tokenizer)
            logger.info("Summarization model loaded")
        except Exception as e:
            logger.warning(f"Summarizer not available: {e}")
            return None
    return _summarizer


class SummarizerWrapper:
    """Wrapper to match the Summarizer protocol."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def summarize(self, text: str, max_length: int) -> str:
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@mcp.tool()
async def compress(
    text: Annotated[str, "Text content to compress"],
    query: Annotated[str, "Query for relevance scoring (what to look for)"] = "",
    budget_tokens: Annotated[int, "Target token budget for output"] = 900,
    file_path: Annotated[str, "Optional file path for type detection"] = "",
    use_cache: Annotated[bool, "Whether to use caching"] = True,
) -> dict:
    """Compress text using the SCOPE algorithm.

    Uses semantic chunking, relevance scoring via embeddings, and dynamic
    compression to reduce token usage while preserving important content.

    Returns compressed text with a symbol index for navigation.
    """
    await _ensure_db()

    # Estimate tokens (rough: 4 chars per token)
    original_tokens = len(text) // 4

    # Check cache first
    text_hash = content_hash(text)
    query_hash = content_hash(query) if query else None

    if use_cache:
        cached = await db.get_compressed(text_hash, query_hash, budget_tokens, MODEL_VERSION)
        if cached:
            return {
                "compressed_text": cached.compressed_text,
                "original_tokens": cached.original_tokens or original_tokens,
                "compressed_tokens": cached.compressed_tokens or len(cached.compressed_text) // 4,
                "compression_ratio": cached.compression_ratio or 0.5,
                "cache_hit": True,
                "model_version": MODEL_VERSION,
            }

    # Determine file type
    file_type = detect_file_type(file_path) if file_path else "unknown"

    # Try to compress with models
    symbol_index = ""
    try:
        embedder = await _get_embedder()

        # Summarizer is optional
        summarizer = None
        summarizer_data = await _get_summarizer()
        if summarizer_data:
            model, tokenizer = summarizer_data
            summarizer = SummarizerWrapper(model, tokenizer)

        compressed, symbol_index = compress_with_scope_indexed(
            text=text,
            query=query,
            budget_tokens=budget_tokens,
            embedder=embedder,
            summarizer=summarizer,
            file_type=file_type,
        )
    except Exception as e:
        logger.warning(f"SCOPE compression failed, using quick fallback: {e}")
        compressed, symbol_index = quick_compress_indexed(text, budget_tokens * 4, file_type)

    compressed_tokens = len(compressed) // 4
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

    # Store in cache
    cache_text = f"{symbol_index}\n\n{compressed}" if symbol_index else compressed
    entry = CacheEntry(
        content_hash=text_hash,
        query_hash=query_hash,
        budget_tokens=budget_tokens,
        model_version=MODEL_VERSION,
        compressed_text=cache_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=ratio,
    )
    await db.put_compressed(entry)

    result = {
        "compressed_text": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": ratio,
        "cache_hit": False,
        "model_version": MODEL_VERSION,
    }
    if symbol_index:
        result["symbol_index"] = symbol_index

    return result


@mcp.tool()
async def compress_quick(
    text: Annotated[str, "Text content to compress"],
    budget_tokens: Annotated[int, "Target token budget for output"] = 900,
    file_path: Annotated[str, "Optional file path for type detection"] = "",
) -> dict:
    """Quick compression without ML models (head+tail truncation).

    Faster than full SCOPE compression but less intelligent. Good for
    situations where latency is critical or models aren't available.
    """
    budget_chars = budget_tokens * 4
    original_tokens = len(text) // 4

    file_type = detect_file_type(file_path) if file_path else "unknown"
    compressed, symbol_index = quick_compress_indexed(text, budget_chars, file_type)
    compressed_tokens = len(compressed) // 4

    result = {
        "compressed_text": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
        "method": "quick",
    }
    if symbol_index:
        result["symbol_index"] = symbol_index

    return result


@mcp.tool()
async def summarize(
    text: Annotated[str, "Text content to summarize"],
    max_tokens: Annotated[int, "Maximum tokens for summary"] = 150,
) -> dict:
    """Summarize text content using the DistilBART model.

    Generates a concise summary of the input text.
    """
    summarizer_data = await _get_summarizer()
    if not summarizer_data:
        # Fallback to quick compress if summarizer not available
        from .scope import quick_compress

        summary = quick_compress(text, max_tokens * 4)
        return {
            "summary": summary,
            "original_tokens": len(text) // 4,
            "summary_tokens": len(summary) // 4,
            "method": "fallback",
        }

    model, tokenizer = summarizer_data
    summarizer = SummarizerWrapper(model, tokenizer)
    summary = summarizer.summarize(text, max_tokens)

    return {
        "summary": summary,
        "original_tokens": len(text) // 4,
        "summary_tokens": len(summary) // 4,
        "method": "distilbart",
    }


@mcp.tool()
async def get_file_summary(
    file_path: Annotated[str, "Path to the file to summarize"],
    content: Annotated[str, "File content (if not provided, file will be read)"] = "",
) -> dict:
    """Get or create a cached summary for a file.

    If a cached summary exists and the file hasn't changed, returns it.
    Otherwise generates a new summary and caches it.
    """
    await _ensure_db()

    # Read file if content not provided
    if not content:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Cannot read file: {e}"}

    file_hash = content_hash(content)

    # Check cache
    cached = await db.get_file_summary(file_path, file_hash, MODEL_VERSION)
    if cached:
        return {
            "summary": cached.summary,
            "summary_tokens": cached.summary_tokens or 0,
            "cache_hit": True,
        }

    # Generate summary
    summarizer_data = await _get_summarizer()
    if summarizer_data:
        model, tokenizer = summarizer_data
        summarizer = SummarizerWrapper(model, tokenizer)
        summary = summarizer.summarize(content[:8000], max_length=200)
    else:
        from .scope import quick_compress

        summary = quick_compress(content, 800)

    summary_tokens = len(summary) // 4

    # Cache it
    await db.put_file_summary(
        FileSummary(
            file_path=file_path,
            file_hash=file_hash,
            summary=summary,
            summary_tokens=summary_tokens,
            file_size=len(content),
            file_type=detect_file_type(file_path),
            model_version=MODEL_VERSION,
        )
    )

    return {
        "summary": summary,
        "summary_tokens": summary_tokens,
        "cache_hit": False,
    }


@mcp.tool()
async def compress_mcp_output(
    content: Annotated[str, "MCP tool output content to compress"],
    query: Annotated[str, "Query for relevance scoring (e.g., original user question)"] = "",
    source: Annotated[str, "Source identifier (e.g., 'context7', 'github')"] = "mcp",
    budget_tokens: Annotated[int, "Target token budget for output"] = 1500,
    threshold_tokens: Annotated[int, "Only compress if content exceeds this"] = 1000,
) -> dict:
    """Compress large MCP tool output for context efficiency.

    Use this to compress results from other MCP tools like Context7,
    GitHub MCP, or any tool that returns large text content.

    Example workflow:
    1. Call mcp__context7__get-library-docs to get documentation
    2. If result is large, call this tool to compress it
    3. Use compressed content in your context

    The query parameter significantly improves compression quality by
    preserving content relevant to what you're looking for.
    """
    original_tokens = len(content) // 4

    # Skip compression if under threshold
    if original_tokens < threshold_tokens:
        return {
            "content": content,
            "compressed": False,
            "original_tokens": original_tokens,
            "reason": "Content under threshold, no compression needed",
        }

    # Use the compress function internally
    result = await compress(
        text=content,
        query=query,
        budget_tokens=budget_tokens,
        use_cache=True,
    )

    # Format output with metadata header
    compressed_text = result["compressed_text"]
    symbol_index = result.get("symbol_index", "")
    compressed_tokens = result["compressed_tokens"]
    ratio = result["compression_ratio"]

    header = f"[{source}: Compressed {original_tokens}→{compressed_tokens} tokens ({ratio:.0%})]"
    if symbol_index:
        formatted = f"{header}\n{symbol_index}\n\n{compressed_text}"
    else:
        formatted = f"{header}\n\n{compressed_text}"

    return {
        "content": formatted,
        "compressed": True,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": ratio,
        "source": source,
        "cache_hit": result.get("cache_hit", False),
    }


@mcp.tool()
async def health() -> dict:
    """Check the health status of the ScopePack MCP server.

    Returns information about loaded models and configuration.
    """
    return {
        "status": "ok",
        "embedder_type": EMBEDDER_TYPE,
        "embedder_loaded": _embedder is not None,
        "summarizer_loaded": _summarizer is not None,
        "model_version": MODEL_VERSION,
        "aws_region": AWS_REGION,
    }


@mcp.resource("scopepack://config")
def get_config() -> str:
    """Get ScopePack configuration as JSON."""
    import json

    return json.dumps(
        {
            "embedder_type": EMBEDDER_TYPE,
            "model_version": MODEL_VERSION,
            "aws_region": AWS_REGION,
            "aws_profile": AWS_PROFILE or "(default)",
        },
        indent=2,
    )


def main():
    """Run the MCP server."""
    transport = "stdio"
    port = 8765

    # Parse simple args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1

    logger.info(f"Starting ScopePack MCP server (transport={transport})")

    if transport == "streamable-http":
        mcp.run(transport="streamable-http", port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
