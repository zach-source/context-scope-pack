"""ScopePack: Token-efficient context management for Claude Code."""

__version__ = "0.1.0"

# Core compression functions
# MCP output compression utilities
from .mcp_utils import (
    MCPResultCompressor,
    compress_context7_result,
    compress_github_result,
    compress_if_large,
    compress_mcp_result,
    compress_web_fetch_result,
)
from .scope import (
    compress_with_scope,
    compress_with_scope_indexed,
    detect_file_type,
    quick_compress,
    quick_compress_indexed,
)

__all__ = [
    # Core
    "compress_with_scope",
    "compress_with_scope_indexed",
    "quick_compress",
    "quick_compress_indexed",
    "detect_file_type",
    # MCP utilities
    "compress_mcp_result",
    "compress_if_large",
    "compress_context7_result",
    "compress_github_result",
    "compress_web_fetch_result",
    "MCPResultCompressor",
]
