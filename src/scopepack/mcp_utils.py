"""Utilities for integrating SCOPE compression with MCP server outputs.

This module provides functions for compressing large MCP tool results
(like Context7 documentation) before adding them to agent context.

Usage:
    # Agent-level post-processing
    from scopepack.mcp_utils import compress_mcp_result, compress_if_large

    # After getting Context7 result
    compressed = compress_mcp_result(
        content=context7_result["content"],
        query="React hooks useState",
        source="context7",
    )
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scope import Embedder

logger = logging.getLogger("scopepack.mcp_utils")

# Default configuration
DEFAULT_TOKEN_THRESHOLD = 1000  # Compress if content exceeds this
DEFAULT_BUDGET_TOKENS = 900
DEFAULT_CHARS_PER_TOKEN = 4

# Daemon URL for HTTP-based compression
DAEMON_URL = "http://127.0.0.1:18765"


def estimate_tokens(text: str, chars_per_token: int = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate token count from text length."""
    return len(text) // chars_per_token


def content_hash(text: str) -> str:
    """Generate short content hash for caching."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compress_if_large(
    content: str,
    query: str = "",
    threshold_tokens: int = DEFAULT_TOKEN_THRESHOLD,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    use_daemon: bool = True,
) -> str:
    """Compress content if it exceeds the token threshold.

    Args:
        content: Text content to potentially compress
        query: Query for relevance scoring (improves compression quality)
        threshold_tokens: Only compress if content exceeds this token count
        budget_tokens: Target token budget for compressed output
        use_daemon: If True, use HTTP daemon; if False, use direct library

    Returns:
        Original content if under threshold, compressed content otherwise
    """
    estimated_tokens = estimate_tokens(content)

    if estimated_tokens < threshold_tokens:
        return content

    if use_daemon:
        return _compress_via_daemon(content, query, budget_tokens)
    else:
        return _compress_direct(content, query, budget_tokens)


def compress_mcp_result(
    content: str | dict[str, Any] | list[Any],
    query: str = "",
    source: str = "mcp",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    threshold_tokens: int = DEFAULT_TOKEN_THRESHOLD,
    use_daemon: bool = True,
    include_metadata: bool = True,
) -> str:
    """Compress MCP tool result content for context efficiency.

    Designed for use after receiving results from MCP tools like Context7,
    GitHub MCP, or other documentation/content providers.

    Args:
        content: MCP result content (string, dict, or list)
        query: Query for relevance scoring (e.g., the original user question)
        source: Source identifier for logging (e.g., "context7", "github")
        budget_tokens: Target token budget for compressed output
        threshold_tokens: Only compress if content exceeds this token count
        use_daemon: If True, use HTTP daemon; if False, use direct library
        include_metadata: If True, include compression stats header

    Returns:
        Compressed content string with optional metadata header

    Example:
        # After calling Context7
        docs = context7_result.get("content", "")
        compressed = compress_mcp_result(
            content=docs,
            query="React hooks useState useEffect",
            source="context7",
        )
    """
    # Normalize content to string
    if isinstance(content, dict):
        text = _dict_to_text(content)
    elif isinstance(content, list):
        text = _list_to_text(content)
    else:
        text = str(content)

    original_tokens = estimate_tokens(text)

    if original_tokens < threshold_tokens:
        logger.debug(
            f"[{source}] Content under threshold ({original_tokens} tokens), skipping compression"
        )
        return text

    logger.info(f"[{source}] Compressing {original_tokens} tokens to ~{budget_tokens}")

    if use_daemon:
        result = _compress_via_daemon_full(text, query, budget_tokens)
    else:
        result = _compress_direct_full(text, query, budget_tokens)

    compressed_text = result["compressed_text"]
    symbol_index = result.get("symbol_index", "")
    compressed_tokens = result.get("compressed_tokens", estimate_tokens(compressed_text))
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

    if include_metadata:
        header = (
            f"[{source}: Compressed {original_tokens}→{compressed_tokens} tokens ({ratio:.0%})]"
        )
        if symbol_index:
            return f"{header}\n{symbol_index}\n\n{compressed_text}"
        return f"{header}\n\n{compressed_text}"

    if symbol_index:
        return f"{symbol_index}\n\n{compressed_text}"
    return compressed_text


def _dict_to_text(d: dict[str, Any], indent: int = 0) -> str:
    """Convert dict to readable text format."""
    lines = []
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_dict_to_text(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            lines.append(_list_to_text(value, indent + 1))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def _list_to_text(lst: list[Any], indent: int = 0) -> str:
    """Convert list to readable text format."""
    lines = []
    prefix = "  " * indent
    for item in lst:
        if isinstance(item, dict):
            lines.append(_dict_to_text(item, indent))
            lines.append("")  # Blank line between items
        elif isinstance(item, list):
            lines.append(_list_to_text(item, indent))
        else:
            lines.append(f"{prefix}- {item}")
    return "\n".join(lines)


def _compress_via_daemon(text: str, query: str, budget_tokens: int) -> str:
    """Compress via HTTP daemon (simple string return)."""
    result = _compress_via_daemon_full(text, query, budget_tokens)
    compressed = result["compressed_text"]
    symbol_index = result.get("symbol_index", "")
    if symbol_index:
        return f"{symbol_index}\n\n{compressed}"
    return compressed


def _compress_via_daemon_full(text: str, query: str, budget_tokens: int) -> dict[str, Any]:
    """Compress via HTTP daemon (full result dict)."""
    try:
        import httpx

        response = httpx.post(
            f"{DAEMON_URL}/compress",
            json={
                "text": text,
                "query": query,
                "budget_tokens": budget_tokens,
                "use_cache": True,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()
    except ImportError:
        logger.warning("httpx not installed, falling back to direct compression")
        return _compress_direct_full(text, query, budget_tokens)
    except Exception as e:
        logger.warning(f"Daemon compression failed: {e}, falling back to direct")
        return _compress_direct_full(text, query, budget_tokens)


def _compress_direct(text: str, query: str, budget_tokens: int) -> str:
    """Compress directly using SCOPE library (simple string return)."""
    result = _compress_direct_full(text, query, budget_tokens)
    compressed = result["compressed_text"]
    symbol_index = result.get("symbol_index", "")
    if symbol_index:
        return f"{symbol_index}\n\n{compressed}"
    return compressed


def _compress_direct_full(text: str, query: str, budget_tokens: int) -> dict[str, Any]:
    """Compress directly using SCOPE library (full result dict)."""
    try:
        from .scope import compress_with_scope_indexed, quick_compress_indexed

        # Try full SCOPE compression first
        try:
            from .embedders import create_embedder

            embedder = create_embedder("bge-small-en-v1.5")
            compressed, symbol_index = compress_with_scope_indexed(
                text=text,
                query=query,
                budget_tokens=budget_tokens,
                embedder=embedder,
                file_type="prose",
            )
        except Exception as e:
            logger.debug(f"Full SCOPE failed, using quick: {e}")
            compressed, symbol_index = quick_compress_indexed(
                text,
                budget_tokens * 4,
                file_type="prose",
            )

        return {
            "compressed_text": compressed,
            "symbol_index": symbol_index,
            "original_tokens": estimate_tokens(text),
            "compressed_tokens": estimate_tokens(compressed),
        }
    except ImportError as e:
        logger.error(f"SCOPE library not available: {e}")
        # Fallback to simple truncation
        budget_chars = budget_tokens * 4
        if len(text) <= budget_chars:
            return {"compressed_text": text, "symbol_index": ""}
        truncated = (
            text[: budget_chars // 2] + "\n\n[...truncated...]\n\n" + text[-budget_chars // 2 :]
        )
        return {"compressed_text": truncated, "symbol_index": ""}


# Convenience functions for specific MCP sources


def compress_context7_result(
    content: str | dict[str, Any],
    topic: str = "",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> str:
    """Compress Context7 library documentation result.

    Args:
        content: Context7 result content
        topic: The topic parameter used in the Context7 query
        budget_tokens: Target token budget

    Example:
        result = context7.get_library_docs("/vercel/next.js", topic="routing")
        compressed = compress_context7_result(result["content"], topic="routing")
    """
    return compress_mcp_result(
        content=content,
        query=topic,
        source="context7",
        budget_tokens=budget_tokens,
    )


def compress_github_result(
    content: str | dict[str, Any],
    query: str = "",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> str:
    """Compress GitHub MCP result (file contents, PR diffs, etc.).

    Args:
        content: GitHub MCP result content
        query: Search query or context for relevance
        budget_tokens: Target token budget
    """
    return compress_mcp_result(
        content=content,
        query=query,
        source="github",
        budget_tokens=budget_tokens,
    )


def compress_web_fetch_result(
    content: str,
    query: str = "",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> str:
    """Compress WebFetch result (web page content).

    Args:
        content: WebFetch result content
        query: The prompt/question used for the fetch
        budget_tokens: Target token budget
    """
    return compress_mcp_result(
        content=content,
        query=query,
        source="webfetch",
        budget_tokens=budget_tokens,
    )


# Agent integration helpers


class MCPResultCompressor:
    """Stateful compressor for use in agent loops.

    Caches embedder for efficiency across multiple compressions.

    Example:
        compressor = MCPResultCompressor(budget_tokens=1500)

        # In agent loop
        for tool_result in tool_results:
            if tool_result["tool"] == "mcp__context7__get-library-docs":
                content = compressor.compress(
                    tool_result["content"],
                    query=user_query,
                )
    """

    def __init__(
        self,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        threshold_tokens: int = DEFAULT_TOKEN_THRESHOLD,
        use_daemon: bool = True,
    ):
        self.budget_tokens = budget_tokens
        self.threshold_tokens = threshold_tokens
        self.use_daemon = use_daemon
        self._embedder: Embedder | None = None

    def compress(
        self,
        content: str | dict[str, Any] | list[Any],
        query: str = "",
        source: str = "mcp",
    ) -> str:
        """Compress MCP result content."""
        return compress_mcp_result(
            content=content,
            query=query,
            source=source,
            budget_tokens=self.budget_tokens,
            threshold_tokens=self.threshold_tokens,
            use_daemon=self.use_daemon,
        )

    def should_compress(self, content: str) -> bool:
        """Check if content should be compressed."""
        return estimate_tokens(content) >= self.threshold_tokens
