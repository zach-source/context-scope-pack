"""SCOPE-style compression algorithm.

Based on the SCOPE paper:
1. Semantic chunking (code blocks, paragraphs, functions)
2. Relevance scoring via embeddings
3. Dynamic compression ratios (compress low-relevance more)
4. Summarization with small models (DistilBART)
5. Reassembly in original order
"""

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Chunk:
    """A semantic chunk of content."""

    text: str
    start_idx: int
    end_idx: int
    chunk_type: str  # "code", "prose", "heading", "blank"
    relevance_score: float = 0.0
    compressed_text: str | None = None


class Embedder(Protocol):
    """Protocol for embedding models."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


class Summarizer(Protocol):
    """Protocol for summarization models."""

    def summarize(self, text: str, max_length: int) -> str:
        """Summarize text to max_length tokens."""
        ...


# Regex patterns for semantic chunking
CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"^([ \t]+\S.*\n)+", re.MULTILINE)
HEADING_PATTERN = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
FUNCTION_PATTERN = re.compile(
    r"^(async\s+)?def\s+\w+\s*\([^)]*\)\s*(?:->\s*\S+)?\s*:", re.MULTILINE
)
CLASS_PATTERN = re.compile(r"^class\s+\w+.*:", re.MULTILINE)
BLANK_LINE_PATTERN = re.compile(r"^\s*$", re.MULTILINE)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25, handling snake_case and camelCase.

    Splits identifiers into component words for better lexical matching.
    """
    # Extract word tokens (preserve case for camelCase detection)
    tokens = re.findall(r"\b\w+\b", text)
    expanded = []
    for token in tokens:
        # Split camelCase BEFORE lowercasing: "calculateTotal" -> "calculate Total"
        parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", token).lower().split()
        # Split snake_case: handled by the regex since _ is not in \w boundary
        for part in parts:
            # Also split on underscores within tokens
            expanded.extend(part.split("_"))
    # Filter empty strings
    return [t for t in expanded if t]


def compute_bm25_scores(chunks: list["Chunk"], query: str) -> list[float]:
    """Compute BM25 lexical relevance scores.

    Uses standard BM25 parameters: k1=1.5, b=0.75
    """
    if not chunks or not query:
        return [0.0] * len(chunks)

    # BM25 parameters
    k1 = 1.5
    b = 0.75

    # Tokenize query and chunks
    query_tokens = set(tokenize_for_bm25(query))
    chunk_token_lists = [tokenize_for_bm25(c.text) for c in chunks]

    # Calculate average document length
    doc_lengths = [len(tokens) for tokens in chunk_token_lists]
    avg_dl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0

    # Build document frequency (DF) for each term
    df: dict[str, int] = {}
    for tokens in chunk_token_lists:
        seen = set(tokens)
        for token in seen:
            df[token] = df.get(token, 0) + 1

    n_docs = len(chunks)
    scores = []

    for i, tokens in enumerate(chunk_token_lists):
        # Term frequency in this document
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        doc_len = doc_lengths[i]
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue

            # IDF: log((N - df + 0.5) / (df + 0.5))
            term_df = df.get(term, 0)
            idf = np.log((n_docs - term_df + 0.5) / (term_df + 0.5) + 1)

            # TF component with length normalization
            term_tf = tf[term]
            tf_component = (term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * doc_len / avg_dl))

            score += idf * tf_component

        scores.append(score)

    return scores


def detect_file_type(path: str) -> str:
    """Detect file type from extension."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    code_exts = {
        "py",
        "js",
        "ts",
        "jsx",
        "tsx",
        "go",
        "rs",
        "java",
        "c",
        "cpp",
        "h",
        "hpp",
        "rb",
        "php",
        "sh",
        "bash",
        "zsh",
        "nix",
        "lua",
        "vim",
        "sql",
    }
    config_exts = {"json", "yaml", "yml", "toml", "ini", "cfg", "conf", "xml"}
    doc_exts = {"md", "rst", "txt", "adoc"}

    if ext in code_exts:
        return "code"
    elif ext in config_exts:
        return "config"
    elif ext in doc_exts:
        return "doc"
    return "unknown"


def chunk_code(text: str) -> list[Chunk]:
    """Chunk code into semantic units (functions, classes, blocks)."""
    chunks = []
    lines = text.split("\n")
    current_chunk_lines = []
    current_start = 0
    in_function_or_class = False
    indent_level = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Check for function or class definition
        if stripped.startswith(("def ", "async def ", "class ")):
            # Save previous chunk if exists
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_idx=current_start,
                            end_idx=i - 1,
                            chunk_type="code",
                        )
                    )
            current_chunk_lines = [line]
            current_start = i
            in_function_or_class = True
            indent_level = len(line) - len(stripped)
            continue

        if in_function_or_class:
            # Check if we've exited the function/class
            if stripped and not line.startswith(" " * (indent_level + 1)):
                if not stripped.startswith((")", "]", "}", "#", '"""', "'''")):
                    # End of function/class
                    chunk_text = "\n".join(current_chunk_lines)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_idx=current_start,
                            end_idx=i - 1,
                            chunk_type="code",
                        )
                    )
                    current_chunk_lines = [line]
                    current_start = i
                    in_function_or_class = False
                    continue

        current_chunk_lines.append(line)

    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_idx=current_start,
                    end_idx=len(lines) - 1,
                    chunk_type="code",
                )
            )

    return chunks


def chunk_prose(text: str) -> list[Chunk]:
    """Chunk prose/markdown into semantic units (headings, paragraphs)."""
    chunks = []
    lines = text.split("\n")
    current_chunk_lines = []
    current_start = 0
    current_type = "prose"

    for i, line in enumerate(lines):
        # Check for heading
        if line.startswith("#"):
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_idx=current_start,
                            end_idx=i - 1,
                            chunk_type=current_type,
                        )
                    )
            current_chunk_lines = [line]
            current_start = i
            current_type = "heading"
            continue

        # Check for code fence
        if line.startswith("```"):
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_idx=current_start,
                            end_idx=i - 1,
                            chunk_type=current_type,
                        )
                    )
            current_chunk_lines = [line]
            current_start = i
            current_type = "code"
            # Find closing fence
            for j in range(i + 1, len(lines)):
                current_chunk_lines.append(lines[j])
                if lines[j].startswith("```"):
                    chunk_text = "\n".join(current_chunk_lines)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_idx=current_start,
                            end_idx=j,
                            chunk_type="code",
                        )
                    )
                    current_chunk_lines = []
                    current_start = j + 1
                    current_type = "prose"
                    break
            continue

        # Check for blank line (paragraph separator)
        if not line.strip():
            if current_chunk_lines and current_type == "prose":
                chunk_text = "\n".join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_idx=current_start,
                            end_idx=i - 1,
                            chunk_type=current_type,
                        )
                    )
                current_chunk_lines = []
                current_start = i + 1
            continue

        current_chunk_lines.append(line)

    # Last chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_idx=current_start,
                    end_idx=len(lines) - 1,
                    chunk_type=current_type,
                )
            )

    return chunks


def chunk_text(text: str, file_type: str = "unknown") -> list[Chunk]:
    """Chunk text based on detected file type."""
    if file_type == "code":
        return chunk_code(text)
    elif file_type in ("doc", "prose", "unknown"):
        return chunk_prose(text)
    else:
        # For config files, treat as code
        return chunk_code(text)


def score_relevance(
    chunks: list[Chunk],
    query: str,
    embedder: Embedder,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> list[Chunk]:
    """Score chunk relevance using hybrid semantic + lexical scoring.

    Combines embedding-based semantic similarity with BM25 lexical matching.

    Args:
        chunks: List of text chunks to score
        query: The query to score against
        embedder: Embedding model for semantic similarity
        alpha: Balance between semantic (1.0) and lexical (0.0). Default 0.5
        temperature: Score sharpening factor. Higher = more spread. Default 2.0

    Returns:
        Chunks with relevance_score set (0.0 to 1.0)
    """
    if not query or not chunks:
        # No query = all chunks equally relevant
        for chunk in chunks:
            chunk.relevance_score = 1.0
        return chunks

    # 1. Compute semantic scores (cosine similarity)
    texts = [query] + [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts)
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]

    semantic_scores = []
    for i, _chunk in enumerate(chunks):
        chunk_emb = chunk_embeddings[i]
        similarity = np.dot(query_embedding, chunk_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb) + 1e-8
        )
        semantic_scores.append(float(similarity))

    # 2. Compute BM25 lexical scores
    bm25_scores = compute_bm25_scores(chunks, query)

    # 3. Normalize both score types to [0, 1]
    def normalize(scores: list[float]) -> list[float]:
        if not scores:
            return scores
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 1e-8:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    norm_semantic = normalize(semantic_scores)
    norm_bm25 = normalize(bm25_scores)

    # 4. Combine scores: alpha * semantic + (1-alpha) * lexical
    combined = [
        alpha * sem + (1 - alpha) * lex for sem, lex in zip(norm_semantic, norm_bm25, strict=False)
    ]

    # 5. Apply temperature sharpening to spread scores
    # Higher temperature = more differentiation between scores
    if temperature > 0:
        # Softmax-style sharpening
        exp_scores = [np.exp(s * temperature) for s in combined]
        total = sum(exp_scores)
        if total > 0:
            sharpened = [e / total for e in exp_scores]
            # Renormalize to [0, 1] range
            combined = normalize(sharpened)

    # 6. Assign final scores
    for chunk, score in zip(chunks, combined, strict=False):
        chunk.relevance_score = score

    return chunks


def compute_compression_ratios(chunks: list[Chunk], target_budget: int) -> list[float]:
    """Compute dynamic compression ratios based on relevance.

    Higher relevance = less compression (ratio closer to 1.0)
    Lower relevance = more compression (ratio closer to 0.1)
    """
    if not chunks:
        return []

    # Estimate current total tokens (rough: 4 chars per token)
    current_tokens = sum(len(c.text) // 4 for c in chunks)

    if current_tokens <= target_budget:
        # No compression needed
        return [1.0] * len(chunks)

    # Normalize relevance scores to [0, 1]
    scores = [c.relevance_score for c in chunks]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score > min_score else 1.0

    normalized = [(s - min_score) / score_range for s in scores]

    # Map to compression ratios: high relevance -> 0.8-1.0, low relevance -> 0.1-0.3
    ratios = [0.1 + 0.7 * n for n in normalized]

    # Adjust ratios to hit target budget
    estimated_output = sum((len(c.text) // 4) * r for c, r in zip(chunks, ratios, strict=False))

    if estimated_output > target_budget:
        # Scale down ratios
        scale = target_budget / estimated_output
        ratios = [max(0.1, r * scale) for r in ratios]

    return ratios


def compress_chunk(
    chunk: Chunk,
    ratio: float,
    summarizer: Summarizer | None,
    query: str = "",
) -> str:
    """Compress a single chunk based on ratio, preserving query-relevant lines.

    ratio = 1.0: keep full text
    ratio < 1.0: summarize to ratio * original_length

    When query is provided, lines containing query keywords are prioritized
    for preservation even if they're in the middle of the chunk.
    """
    if ratio >= 0.95:
        return chunk.text

    # Target length in characters
    target_len = int(len(chunk.text) * ratio)

    if summarizer is None:
        # Smart truncation: preserve query-relevant lines
        if query and target_len >= 100:
            result = _smart_truncate(chunk.text, target_len, query)
            if result:
                return result

        # Fallback: head + tail truncation
        if target_len < 100:
            return chunk.text[:target_len] + "..."
        head_len = int(target_len * 0.7)
        tail_len = target_len - head_len - 10
        return chunk.text[:head_len] + "\n...[snip]...\n" + chunk.text[-tail_len:]

    # Use summarizer
    # Estimate max tokens (4 chars per token)
    max_tokens = max(20, target_len // 4)
    return summarizer.summarize(chunk.text, max_length=max_tokens)


def _smart_truncate(text: str, target_len: int, query: str) -> str | None:
    """Truncate text while preserving lines that match query keywords.

    Strategy:
    1. Keep first line (often function signature)
    2. Find and preserve lines containing query keywords
    3. Fill remaining budget with context around keyword lines

    Returns None if no keyword matches found (fall back to regular truncation).
    """
    lines = text.split("\n")
    if len(lines) < 3:
        return None

    # Get query keywords (lowercase, split on whitespace and punctuation)
    keywords = set(tokenize_for_bm25(query))
    if not keywords:
        return None

    # Find lines that contain query keywords
    keyword_line_indices = []
    for i, line in enumerate(lines):
        line_tokens = set(tokenize_for_bm25(line))
        if keywords & line_tokens:  # Intersection
            keyword_line_indices.append(i)

    if not keyword_line_indices:
        return None

    # Build output: first line + keyword lines with context
    output_lines = []
    used_indices = set()

    # Always include first line (signature/header)
    output_lines.append(lines[0])
    used_indices.add(0)
    current_len = len(lines[0]) + 1

    # Add each keyword line with 1 line of context before/after
    for idx in keyword_line_indices:
        # Context window: 1 line before, the line, 1 line after
        start = max(0, idx - 1)
        end = min(len(lines), idx + 2)

        for i in range(start, end):
            if i not in used_indices:
                line_len = len(lines[i]) + 1
                if current_len + line_len <= target_len:
                    used_indices.add(i)
                    current_len += line_len

    # If we have budget left, add more lines from the beginning
    for i, line in enumerate(lines):
        if i not in used_indices:
            line_len = len(line) + 1
            if current_len + line_len <= target_len - 20:  # Reserve space for markers
                used_indices.add(i)
                current_len += line_len

    # Reconstruct with sorted indices and gap markers
    sorted_indices = sorted(used_indices)
    result_parts = []
    prev_idx = -1

    for idx in sorted_indices:
        if prev_idx >= 0 and idx > prev_idx + 1:
            # Gap - add marker
            result_parts.append("    # ... snip ...")
        result_parts.append(lines[idx])
        prev_idx = idx

    # Add final marker if we didn't reach the end
    if sorted_indices and sorted_indices[-1] < len(lines) - 1:
        result_parts.append("    # ... snip ...")

    return "\n".join(result_parts)


def compress_with_scope(
    text: str,
    query: str,
    budget_tokens: int,
    embedder: Embedder,
    summarizer: Summarizer | None = None,
    file_type: str = "unknown",
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> str:
    """Full SCOPE compression pipeline.

    1. Chunk text semantically
    2. Score chunks for relevance to query (hybrid semantic + lexical)
    3. Compute compression ratios
    4. Compress each chunk
    5. Reassemble in order

    Args:
        alpha: Balance between semantic (1.0) and lexical (0.0). Default 0.5
        temperature: Score sharpening factor. Higher = more spread. Default 2.0
    """
    # 1. Chunk
    chunks = chunk_text(text, file_type)

    if not chunks:
        return text[: budget_tokens * 4]  # Fallback

    # 2. Score relevance (hybrid)
    chunks = score_relevance(chunks, query, embedder, alpha=alpha, temperature=temperature)

    # 3. Compute ratios
    ratios = compute_compression_ratios(chunks, budget_tokens)

    # 4. Compress each chunk (pass query for smart truncation)
    compressed_parts = []
    for chunk, ratio in zip(chunks, ratios, strict=False):
        compressed = compress_chunk(chunk, ratio, summarizer, query=query)
        chunk.compressed_text = compressed
        compressed_parts.append(compressed)

    # 5. Reassemble
    return "\n\n".join(compressed_parts)


def quick_compress(text: str, budget_chars: int) -> str:
    """Quick compression without models (head + tail fallback)."""
    if len(text) <= budget_chars:
        return text

    # Keep first 70% of budget, last 20%, 10% for separator
    head_budget = int(budget_chars * 0.7)
    tail_budget = int(budget_chars * 0.2)

    head = text[:head_budget]
    tail = text[-tail_budget:]

    return f"{head}\n\n...[{len(text) - head_budget - tail_budget} chars snipped]...\n\n{tail}"


def extract_symbol_name(text: str, _chunk_type: str) -> str | None:
    """Extract the primary symbol name from a chunk (function, class, heading)."""
    lines = text.strip().split("\n")
    if not lines:
        return None

    first_line = lines[0].strip()

    # Python function
    if first_line.startswith("def ") or first_line.startswith("async def "):
        match = re.match(r"(?:async\s+)?def\s+(\w+)", first_line)
        if match:
            return match.group(1)

    # Python class
    if first_line.startswith("class "):
        match = re.match(r"class\s+(\w+)", first_line)
        if match:
            return match.group(1)

    # Markdown heading
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()

    # JavaScript/TypeScript function
    match = re.match(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", first_line)
    if match:
        return match.group(1)

    # JavaScript/TypeScript const/let arrow function
    match = re.match(r"(?:export\s+)?(?:const|let)\s+(\w+)\s*=", first_line)
    if match:
        return match.group(1)

    # Go function
    match = re.match(r"func\s+(?:\([^)]+\)\s+)?(\w+)", first_line)
    if match:
        return match.group(1)

    # Rust function
    match = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", first_line)
    if match:
        return match.group(1)

    return None


def build_symbol_index(chunks: list[Chunk]) -> str:
    """Build a symbol index with line numbers for navigation.

    Returns a compact index like:
    ## Symbol Index
    - L15-42: class ShoppingCart
    - L44-52: def calculate_total
    - L54-61: def apply_discount (high relevance)
    """
    if not chunks:
        return ""

    index_lines = ["## Symbol Index (use line numbers to request specific sections)"]

    for chunk in chunks:
        symbol = extract_symbol_name(chunk.text, chunk.chunk_type)
        start = chunk.start_idx + 1  # 1-indexed for user display
        end = chunk.end_idx + 1

        # Format line range
        line_ref = f"L{start}" if start == end else f"L{start}-{end}"

        # Include relevance hint for high-relevance chunks
        relevance_hint = ""
        if chunk.relevance_score >= 0.7:
            relevance_hint = " [highly relevant]"
        elif chunk.relevance_score <= 0.3:
            relevance_hint = " [low relevance]"

        if symbol:
            index_lines.append(f"- {line_ref}: {chunk.chunk_type} {symbol}{relevance_hint}")
        else:
            # For chunks without clear symbols, show a preview
            preview = chunk.text[:40].replace("\n", " ").strip()
            if len(chunk.text) > 40:
                preview += "..."
            index_lines.append(f"- {line_ref}: {chunk.chunk_type} `{preview}`{relevance_hint}")

    return "\n".join(index_lines)


def compress_with_scope_indexed(
    text: str,
    query: str,
    budget_tokens: int,
    embedder: Embedder,
    summarizer: Summarizer | None = None,
    file_type: str = "unknown",
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> tuple[str, str]:
    """SCOPE compression with symbol index.

    Returns (compressed_text, symbol_index) tuple.
    The symbol index provides line number references for navigation.

    Args:
        alpha: Balance between semantic (1.0) and lexical (0.0). Default 0.5
        temperature: Score sharpening factor. Higher = more spread. Default 2.0
    """
    # 1. Chunk
    chunks = chunk_text(text, file_type)

    if not chunks:
        return text[: budget_tokens * 4], ""

    # 2. Score relevance (hybrid)
    chunks = score_relevance(chunks, query, embedder, alpha=alpha, temperature=temperature)

    # 3. Build symbol index BEFORE compression (preserves all symbols)
    symbol_index = build_symbol_index(chunks)

    # 4. Compute ratios
    ratios = compute_compression_ratios(chunks, budget_tokens)

    # 5. Compress each chunk with line number prefix
    compressed_parts = []
    for chunk, ratio in zip(chunks, ratios, strict=False):
        # Add line number reference to each chunk
        start = chunk.start_idx + 1
        end = chunk.end_idx + 1
        line_ref = f"L{start}" if start == end else f"L{start}-{end}"

        compressed = compress_chunk(chunk, ratio, summarizer, query=query)
        chunk.compressed_text = compressed

        # Prefix with line reference
        compressed_parts.append(f"[{line_ref}]\n{compressed}")

    # 6. Reassemble
    compressed_text = "\n\n".join(compressed_parts)

    return compressed_text, symbol_index


def quick_compress_indexed(
    text: str, budget_chars: int, file_type: str = "unknown"
) -> tuple[str, str]:
    """Quick compression with symbol index (no ML models needed).

    Returns (compressed_text, symbol_index) tuple.
    """
    # Parse into chunks for symbol extraction
    chunks = chunk_text(text, file_type)

    # Build symbol index from all chunks
    # Give all chunks equal relevance for quick mode
    for chunk in chunks:
        chunk.relevance_score = 0.5
    symbol_index = build_symbol_index(chunks)

    # Do quick compression on the raw text
    if len(text) <= budget_chars:
        compressed = text
    else:
        head_budget = int(budget_chars * 0.7)
        tail_budget = int(budget_chars * 0.2)
        head = text[:head_budget]
        tail = text[-tail_budget:]
        snipped = len(text) - head_budget - tail_budget
        compressed = (
            f"{head}\n\n...[{snipped} chars snipped, see Symbol Index for navigation]...\n\n{tail}"
        )

    return compressed, symbol_index
