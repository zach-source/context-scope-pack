"""AST-based code chunking using tree-sitter.

Uses tree-sitter-language-pack for multi-language AST parsing with a
recursive split-then-merge algorithm for semantic chunking.

Benefits over regex-based chunking:
- Proper nested scope detection (inner functions stay together)
- Decorators attached to their functions
- Multi-line signatures handled correctly
- Language-specific syntax awareness

Requires optional dependency: pip install scopepack[ast]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Parser, Tree

from .scope import Chunk

# Extension to tree-sitter language mapping
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    "py": "python",
    "pyi": "python",
    # JavaScript/TypeScript
    "js": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "mts": "typescript",
    "cts": "typescript",
    "tsx": "tsx",
    # Systems languages
    "go": "go",
    "rs": "rust",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "hpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    # JVM
    "java": "java",
    "kt": "kotlin",
    "kts": "kotlin",
    "scala": "scala",
    # Scripting
    "rb": "ruby",
    "php": "php",
    "lua": "lua",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    # Elixir/Erlang
    "ex": "elixir",
    "exs": "elixir",
    # Haskell
    "hs": "haskell",
    # Config/Data
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    # Markup
    "html": "html",
    "htm": "html",
    "css": "css",
    # Other
    "swift": "swift",
    "nix": "nix",
    "sql": "sql",
}

# Node types that represent "definition" boundaries (split points)
DEFINITION_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "async_function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "export_statement",
        "lexical_declaration",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "export_statement",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "export_statement",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
    "ruby": {
        "method",
        "class",
        "module",
    },
}


@dataclass
class ASTChunkConfig:
    """Configuration for AST-based chunking."""

    min_chunk_chars: int = 100  # Merge chunks smaller than this
    max_chunk_chars: int = 2000  # Split chunks larger than this
    include_scope_prefix: bool = True  # Add "# Scope: ClassName > method_name"


def detect_language(file_path: str) -> str | None:
    """Detect tree-sitter language from file extension.

    Args:
        file_path: Path to the file (or just filename)

    Returns:
        Language name for tree-sitter, or None if not supported
    """
    if "." not in file_path:
        return None

    ext = file_path.rsplit(".", 1)[-1].lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def _get_parser(language: str) -> Parser:
    """Get a tree-sitter parser for the given language.

    Lazily imports tree_sitter_language_pack to avoid import overhead
    when AST chunking isn't used.
    """
    import tree_sitter_language_pack as ts_pack

    return ts_pack.get_parser(language)


def _get_node_text(node: Node, source_bytes: bytes) -> str:
    """Extract text for a node from source bytes."""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _get_line_range(node: Node) -> tuple[int, int]:
    """Get (start_line, end_line) for a node (0-indexed)."""
    return node.start_point[0], node.end_point[0]


def _extract_name(node: Node, source_bytes: bytes, language: str) -> str | None:
    """Extract the name identifier from a definition node."""
    # Common patterns for finding the name child
    name_fields = ["name", "identifier"]

    for field in name_fields:
        name_node = node.child_by_field_name(field)
        if name_node:
            return _get_node_text(name_node, source_bytes)

    # Python decorated_definition: look inside
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in (
                "function_definition",
                "async_function_definition",
                "class_definition",
            ):
                return _extract_name(child, source_bytes, language)

    # Fallback: look for identifier child
    for child in node.children:
        if child.type == "identifier":
            return _get_node_text(child, source_bytes)

    return None


def _build_scope_path(node: Node, source_bytes: bytes, language: str) -> list[str]:
    """Build scope path from root to this node.

    Returns list like ["ClassName", "method_name"] for nested definitions.
    """
    path = []
    current = node

    while current is not None:
        definition_types = DEFINITION_NODE_TYPES.get(language, set())
        if current.type in definition_types:
            name = _extract_name(current, source_bytes, language)
            if name:
                path.append(name)
        current = current.parent

    return list(reversed(path))


def _is_definition_node(node: Node, language: str) -> bool:
    """Check if this node is a definition boundary."""
    definition_types = DEFINITION_NODE_TYPES.get(language, set())
    return node.type in definition_types


def _collect_chunks_recursive(
    node: Node,
    source_bytes: bytes,
    source_lines: list[str],
    language: str,
    config: ASTChunkConfig,
    chunks: list[Chunk],
    scope_path: list[str],
) -> None:
    """Recursively collect chunks using split-then-merge algorithm.

    Strategy:
    1. If node is a definition and fits budget -> create chunk
    2. If node is too large -> recurse into children
    3. Collect non-definition siblings together
    """
    definition_types = DEFINITION_NODE_TYPES.get(language, set())

    if _is_definition_node(node, language):
        node_text = _get_node_text(node, source_bytes)
        node_chars = len(node_text)
        start_line, end_line = _get_line_range(node)

        # Build current scope path
        current_scope = scope_path.copy()
        name = _extract_name(node, source_bytes, language)
        if name:
            current_scope.append(name)

        if node_chars <= config.max_chunk_chars:
            # Node fits budget - create chunk
            text = node_text
            if config.include_scope_prefix and current_scope:
                scope_comment = f"# Scope: {' > '.join(current_scope)}\n"
                text = scope_comment + text

            chunks.append(
                Chunk(
                    text=text,
                    start_idx=start_line,
                    end_idx=end_line,
                    chunk_type="code",
                )
            )
        else:
            # Too large - recurse into children, keeping definition header
            # Find the body child (usually named "body" or "block")
            body_node = node.child_by_field_name("body") or node.child_by_field_name("block")

            if body_node:
                # Create chunk for the header (signature + decorators)
                header_end = body_node.start_point[0] - 1
                if header_end >= start_line:
                    header_lines = source_lines[start_line : header_end + 1]
                    header_text = "\n".join(header_lines)
                    if config.include_scope_prefix and current_scope:
                        scope_comment = f"# Scope: {' > '.join(current_scope)}\n"
                        header_text = scope_comment + header_text

                    if header_text.strip():
                        chunks.append(
                            Chunk(
                                text=header_text,
                                start_idx=start_line,
                                end_idx=header_end,
                                chunk_type="code",
                            )
                        )

                # Recurse into body children
                for child in body_node.children:
                    _collect_chunks_recursive(
                        child, source_bytes, source_lines, language, config, chunks, current_scope
                    )
            else:
                # No body found - recurse into all children
                for child in node.children:
                    if child.type in definition_types:
                        _collect_chunks_recursive(
                            child,
                            source_bytes,
                            source_lines,
                            language,
                            config,
                            chunks,
                            current_scope,
                        )
    else:
        # Not a definition - check if we have definition children
        has_definition_children = any(
            _is_definition_node(child, language) for child in node.children
        )

        if has_definition_children:
            # Process children, grouping non-definitions together
            current_group_start = None
            current_group_end = None

            for child in node.children:
                if _is_definition_node(child, language):
                    # Flush accumulated non-definition group
                    if current_group_start is not None:
                        group_lines = source_lines[current_group_start : current_group_end + 1]
                        group_text = "\n".join(group_lines)
                        if group_text.strip() and len(group_text) >= config.min_chunk_chars:
                            chunks.append(
                                Chunk(
                                    text=group_text,
                                    start_idx=current_group_start,
                                    end_idx=current_group_end,
                                    chunk_type="code",
                                )
                            )
                        current_group_start = None
                        current_group_end = None

                    # Process definition
                    _collect_chunks_recursive(
                        child, source_bytes, source_lines, language, config, chunks, scope_path
                    )
                else:
                    # Accumulate non-definition
                    child_start, child_end = _get_line_range(child)
                    if current_group_start is None:
                        current_group_start = child_start
                    current_group_end = child_end

            # Flush final group
            if current_group_start is not None:
                group_lines = source_lines[current_group_start : current_group_end + 1]
                group_text = "\n".join(group_lines)
                if group_text.strip() and len(group_text) >= config.min_chunk_chars:
                    chunks.append(
                        Chunk(
                            text=group_text,
                            start_idx=current_group_start,
                            end_idx=current_group_end,
                            chunk_type="code",
                        )
                    )


def _merge_small_chunks(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Merge consecutive small chunks below min_chars threshold."""
    if not chunks:
        return chunks

    # Sort by start index
    sorted_chunks = sorted(chunks, key=lambda c: c.start_idx)
    merged = []
    current = sorted_chunks[0]

    for next_chunk in sorted_chunks[1:]:
        # Check if current chunk is small and adjacent/overlapping with next
        is_small = len(current.text) < min_chars
        is_adjacent = next_chunk.start_idx <= current.end_idx + 2  # Allow 1 line gap

        if is_small and is_adjacent:
            # Merge: combine text and extend range
            combined_text = current.text + "\n" + next_chunk.text
            current = Chunk(
                text=combined_text,
                start_idx=current.start_idx,
                end_idx=max(current.end_idx, next_chunk.end_idx),
                chunk_type=current.chunk_type,
            )
        else:
            merged.append(current)
            current = next_chunk

    merged.append(current)
    return merged


def chunk_with_ast(
    text: str,
    language: str,
    config: ASTChunkConfig | None = None,
) -> list[Chunk]:
    """Chunk code using AST-based semantic analysis.

    Uses tree-sitter to parse the code and extract semantic boundaries
    (functions, classes, methods) as chunk points.

    Args:
        text: Source code to chunk
        language: Tree-sitter language name (e.g., "python", "javascript")
        config: Chunking configuration, or None for defaults

    Returns:
        List of Chunk objects with proper boundaries

    Raises:
        ImportError: If tree-sitter-language-pack is not installed
        ValueError: If language is not supported
    """
    if config is None:
        config = ASTChunkConfig()

    try:
        parser = _get_parser(language)
    except ImportError:
        raise ImportError(
            "tree-sitter-language-pack is required for AST chunking. "
            "Install with: pip install scopepack[ast]"
        ) from None
    except Exception as e:
        raise ValueError(f"Language '{language}' is not supported: {e}") from e

    # Parse the source
    source_bytes = text.encode("utf-8")
    source_lines = text.split("\n")
    tree: Tree = parser.parse(source_bytes)
    root = tree.root_node

    # Check for parse errors
    if root.has_error:
        # Still try to extract what we can
        pass

    chunks: list[Chunk] = []

    # Process top-level children
    _collect_chunks_recursive(
        root, source_bytes, source_lines, language, config, chunks, scope_path=[]
    )

    # If no chunks extracted, fall back to whole file
    if not chunks:
        return [
            Chunk(
                text=text,
                start_idx=0,
                end_idx=len(source_lines) - 1,
                chunk_type="code",
            )
        ]

    # Merge small chunks
    chunks = _merge_small_chunks(chunks, config.min_chunk_chars)

    # Sort by start line
    chunks.sort(key=lambda c: c.start_idx)

    return chunks
