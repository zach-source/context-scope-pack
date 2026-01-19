"""Tests for SCOPE compression algorithm."""

import numpy as np

from scopepack.scope import (
    Chunk,
    _smart_truncate,
    chunk_code,
    chunk_prose,
    compress_chunk,
    compute_bm25_scores,
    compute_compression_ratios,
    detect_file_type,
    quick_compress,
    score_relevance,
    tokenize_for_bm25,
)


class TestFileTypeDetection:
    def test_python_file(self):
        assert detect_file_type("main.py") == "code"
        assert detect_file_type("/path/to/script.py") == "code"

    def test_javascript_file(self):
        assert detect_file_type("app.js") == "code"
        assert detect_file_type("component.tsx") == "code"

    def test_config_file(self):
        assert detect_file_type("config.json") == "config"
        assert detect_file_type("settings.yaml") == "config"
        assert detect_file_type("pyproject.toml") == "config"

    def test_doc_file(self):
        assert detect_file_type("README.md") == "doc"
        assert detect_file_type("notes.txt") == "doc"

    def test_unknown_file(self):
        assert detect_file_type("noextension") == "unknown"
        assert detect_file_type("file.xyz") == "unknown"


class TestCodeChunking:
    def test_simple_function(self):
        code = """def hello():
    print("hello")

def world():
    print("world")
"""
        chunks = chunk_code(code)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_class_with_methods(self):
        code = """class MyClass:
    def __init__(self):
        self.x = 1

    def method(self):
        return self.x
"""
        chunks = chunk_code(code)
        assert len(chunks) >= 1

    def test_empty_code(self):
        chunks = chunk_code("")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_code("   \n\n   ")
        assert chunks == []


class TestProseChunking:
    def test_markdown_headings(self):
        text = """# Heading 1

Some content here.

## Heading 2

More content.
"""
        chunks = chunk_prose(text)
        assert len(chunks) >= 2

    def test_code_fences(self):
        text = """Some text

```python
print("hello")
```

More text
"""
        chunks = chunk_prose(text)
        # Should have prose, code fence, prose
        code_chunks = [c for c in chunks if c.chunk_type == "code"]
        assert len(code_chunks) >= 1

    def test_empty_prose(self):
        chunks = chunk_prose("")
        assert chunks == []


class TestCompressionRatios:
    def test_within_budget_no_compression(self):
        chunks = [
            Chunk(
                text="short text",
                start_idx=0,
                end_idx=1,
                chunk_type="prose",
                relevance_score=0.5,
            )
        ]
        ratios = compute_compression_ratios(chunks, target_budget=1000)
        # Should be no compression needed
        assert all(r >= 0.9 for r in ratios)

    def test_high_relevance_less_compression(self):
        chunks = [
            Chunk(
                text="x" * 1000,
                start_idx=0,
                end_idx=100,
                chunk_type="prose",
                relevance_score=0.9,
            ),
            Chunk(
                text="y" * 1000,
                start_idx=101,
                end_idx=200,
                chunk_type="prose",
                relevance_score=0.1,
            ),
        ]
        ratios = compute_compression_ratios(chunks, target_budget=100)
        # High relevance should have higher ratio
        assert ratios[0] > ratios[1]

    def test_empty_chunks(self):
        ratios = compute_compression_ratios([], target_budget=100)
        assert ratios == []


class TestQuickCompress:
    def test_small_text_unchanged(self):
        text = "small text"
        result = quick_compress(text, budget_chars=100)
        assert result == text

    def test_large_text_compressed(self):
        text = "x" * 10000
        result = quick_compress(text, budget_chars=1000)
        assert len(result) < len(text)
        assert "snipped" in result

    def test_preserves_head_and_tail(self):
        text = "HEAD" + "x" * 10000 + "TAIL"
        result = quick_compress(text, budget_chars=1000)
        assert result.startswith("HEAD")
        assert result.endswith("TAIL")


class TestBM25Tokenization:
    def test_simple_words(self):
        tokens = tokenize_for_bm25("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_snake_case_splitting(self):
        tokens = tokenize_for_bm25("score_relevance compute_bm25")
        assert "score" in tokens
        assert "relevance" in tokens
        assert "compute" in tokens
        assert "bm25" in tokens

    def test_camel_case_splitting(self):
        tokens = tokenize_for_bm25("scoreRelevance computeBM25")
        assert "score" in tokens
        assert "relevance" in tokens
        # camelCase splitting should work
        assert "compute" in tokens

    def test_mixed_identifiers(self):
        tokens = tokenize_for_bm25("def calculate_totalCost(items):")
        assert "calculate" in tokens
        assert "total" in tokens
        assert "cost" in tokens
        assert "items" in tokens


class TestBM25Scoring:
    def test_exact_match_boost(self):
        """Chunks containing query terms should score higher."""
        chunks = [
            Chunk(
                text="def score_relevance(chunks, query): pass",
                start_idx=0,
                end_idx=1,
                chunk_type="code",
            ),
            Chunk(
                text="def unrelated_function(): return None",
                start_idx=2,
                end_idx=3,
                chunk_type="code",
            ),
        ]
        scores = compute_bm25_scores(chunks, "score relevance")
        # First chunk has the query terms
        assert scores[0] > scores[1]

    def test_empty_query(self):
        chunks = [Chunk(text="some code", start_idx=0, end_idx=1, chunk_type="code")]
        scores = compute_bm25_scores(chunks, "")
        assert scores == [0.0]

    def test_empty_chunks(self):
        scores = compute_bm25_scores([], "query")
        assert scores == []

    def test_no_matching_terms(self):
        chunks = [
            Chunk(
                text="completely different content",
                start_idx=0,
                end_idx=1,
                chunk_type="code",
            )
        ]
        scores = compute_bm25_scores(chunks, "xyz abc")
        assert scores[0] == 0.0


class TestHybridScoring:
    """Tests for hybrid semantic + lexical scoring."""

    class MockEmbedder:
        """Mock embedder that returns predictable embeddings."""

        def encode(self, texts: list[str]) -> np.ndarray:
            # Return embeddings based on text length for predictability
            embeddings = []
            for text in texts:
                # Create a simple embedding based on character codes
                vec = np.zeros(64)
                for i, char in enumerate(text[:64]):
                    vec[i] = ord(char) / 255.0
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec)
            return np.array(embeddings)

    def test_hybrid_combines_signals(self):
        """Hybrid scoring should use both semantic and lexical signals."""
        chunks = [
            Chunk(
                text="def score_relevance(chunks, query, embedder): pass",
                start_idx=0,
                end_idx=1,
                chunk_type="code",
            ),
            Chunk(
                text="def unrelated_helper(): return 42",
                start_idx=2,
                end_idx=3,
                chunk_type="code",
            ),
            Chunk(
                text="def another_function(): return None",
                start_idx=4,
                end_idx=5,
                chunk_type="code",
            ),
        ]
        embedder = self.MockEmbedder()

        # Score with hybrid (alpha=0.5) - use temperature=0 to avoid normalization issues
        scored = score_relevance(
            chunks, "score relevance query", embedder, alpha=0.5, temperature=0
        )

        # First chunk should score highest (has exact query terms)
        assert scored[0].relevance_score >= scored[1].relevance_score
        assert scored[0].relevance_score >= scored[2].relevance_score

    def test_alpha_controls_balance(self):
        """Alpha=0 should be pure lexical, alpha=1 should be pure semantic."""
        # Test that alpha=0 (pure lexical) correctly boosts exact keyword matches
        chunks = [
            Chunk(
                text="def cosine_similarity(): pass",
                start_idx=0,
                end_idx=1,
                chunk_type="code",
            ),
            Chunk(
                text="def vector_dot_product(): pass",
                start_idx=2,
                end_idx=3,
                chunk_type="code",
            ),
            Chunk(
                text="def unrelated_function(): return None",
                start_idx=4,
                end_idx=5,
                chunk_type="code",
            ),
        ]
        embedder = self.MockEmbedder()

        # Pure lexical (alpha=0) - first chunk should score highest for "cosine"
        chunks_lexical = [
            Chunk(
                text=c.text,
                start_idx=c.start_idx,
                end_idx=c.end_idx,
                chunk_type=c.chunk_type,
            )
            for c in chunks
        ]
        score_relevance(chunks_lexical, "cosine", embedder, alpha=0.0, temperature=0)

        # With pure lexical scoring, "cosine" should be in first chunk only
        # First chunk should have highest score
        assert chunks_lexical[0].relevance_score == max(c.relevance_score for c in chunks_lexical)

    def test_temperature_spreads_scores(self):
        """Higher temperature should increase differentiation between scores."""
        chunks = [
            Chunk(text="def func_a(): pass", start_idx=0, end_idx=1, chunk_type="code"),
            Chunk(text="def func_b(): pass", start_idx=2, end_idx=3, chunk_type="code"),
            Chunk(text="def func_c(): pass", start_idx=4, end_idx=5, chunk_type="code"),
        ]
        embedder = self.MockEmbedder()

        # Low temperature
        chunks_low = [
            Chunk(
                text=c.text,
                start_idx=c.start_idx,
                end_idx=c.end_idx,
                chunk_type=c.chunk_type,
            )
            for c in chunks
        ]
        score_relevance(chunks_low, "func_a", embedder, alpha=0.5, temperature=0.5)

        # High temperature
        chunks_high = [
            Chunk(
                text=c.text,
                start_idx=c.start_idx,
                end_idx=c.end_idx,
                chunk_type=c.chunk_type,
            )
            for c in chunks
        ]
        score_relevance(chunks_high, "func_a", embedder, alpha=0.5, temperature=5.0)

        # Compute score spread (max - min)
        spread_low = max(c.relevance_score for c in chunks_low) - min(
            c.relevance_score for c in chunks_low
        )
        spread_high = max(c.relevance_score for c in chunks_high) - min(
            c.relevance_score for c in chunks_high
        )

        # Higher temperature should create more spread
        assert spread_high >= spread_low

    def test_no_query_returns_equal_scores(self):
        """Without a query, all chunks should have equal relevance."""
        chunks = [
            Chunk(text="def func_a(): pass", start_idx=0, end_idx=1, chunk_type="code"),
            Chunk(text="def func_b(): pass", start_idx=2, end_idx=3, chunk_type="code"),
        ]
        embedder = self.MockEmbedder()

        scored = score_relevance(chunks, "", embedder)
        assert all(c.relevance_score == 1.0 for c in scored)


class TestSmartTruncate:
    """Tests for keyword-aware smart truncation."""

    def test_preserves_lines_with_query_keywords(self):
        """Lines containing query keywords should be preserved."""
        text = """def score_relevance(chunks, query, embedder):
    # Calculate semantic scores
    semantic_scores = []
    for chunk in chunks:
        similarity = compute_cosine(chunk, query)
        semantic_scores.append(similarity)
    # Calculate BM25 lexical scores
    bm25_scores = compute_bm25(chunks, query)
    # Combine scores
    combined = alpha * semantic + (1-alpha) * lexical
    return combined"""

        # Target length that forces truncation but allows keyword lines
        result = _smart_truncate(text, target_len=250, query="cosine similarity semantic")

        assert result is not None
        # Should preserve first line (signature)
        assert "def score_relevance" in result
        # Should preserve line with "cosine"
        assert "cosine" in result
        # Should preserve line with "semantic"
        assert "semantic" in result

    def test_returns_none_when_no_keywords_match(self):
        """Should return None when no query keywords match (fall back to regular truncation)."""
        text = """def hello():
    print("hello world")
    return True"""

        result = _smart_truncate(text, target_len=100, query="xyz nonexistent terms")
        assert result is None

    def test_includes_context_around_keyword_lines(self):
        """Should include 1 line of context before/after keyword matches."""
        text = """def process_data():
    data = load_data()
    cleaned = clean_data(data)
    embeddings = embedder.encode(cleaned)
    scores = compute_scores(embeddings)
    return scores"""

        result = _smart_truncate(text, target_len=250, query="embedder encode")

        assert result is not None
        # Should have the signature
        assert "def process_data" in result
        # Should have the keyword line
        assert "embedder.encode" in result
        # Should have context (line before: cleaned = ...)
        assert "cleaned" in result

    def test_handles_short_text(self):
        """Should return None for text with fewer than 3 lines."""
        text = "def foo(): pass"
        result = _smart_truncate(text, target_len=100, query="foo")
        assert result is None

    def test_handles_empty_query(self):
        """Should return None for empty query."""
        text = """def foo():
    x = 1
    y = 2
    return x + y"""
        result = _smart_truncate(text, target_len=100, query="")
        assert result is None

    def test_adds_snip_markers_for_gaps(self):
        """Should add snip markers when there are gaps in preserved lines."""
        text = """def calculate_total(items):
    subtotal = 0
    for item in items:
        subtotal += item.price
    discount = apply_discount(subtotal)
    tax = calculate_tax(subtotal - discount)
    final_total = subtotal - discount + tax
    return final_total"""

        # Query that matches lines at beginning and end
        result = _smart_truncate(text, target_len=180, query="calculate_total final_total")

        assert result is not None
        # Should have snip marker
        assert "snip" in result.lower()


class TestCompressChunk:
    """Tests for compress_chunk with query-aware truncation."""

    def test_uses_smart_truncate_when_query_provided(self):
        """compress_chunk should use smart truncation when query is provided."""
        # Larger chunk to allow meaningful smart truncation
        chunk = Chunk(
            text="""def score_relevance(chunks, query, embedder):
    '''Score chunks for relevance to query.'''
    # Initialize containers
    semantic_scores = []
    bm25_scores = []

    # Calculate semantic similarity using embeddings
    for chunk in chunks:
        embedding = embedder.encode([chunk.text])
        similarity = compute_cosine(embedding, query_embedding)
        semantic_scores.append(similarity)

    # Calculate BM25 lexical scores for keyword matching
    for chunk in chunks:
        score = compute_bm25(chunk.text, query)
        bm25_scores.append(score)

    # Combine semantic and lexical signals
    combined = []
    for sem, lex in zip(semantic_scores, bm25_scores):
        combined.append(alpha * sem + (1-alpha) * lex)

    return combined""",
            start_idx=0,
            end_idx=25,
            chunk_type="code",
            relevance_score=0.5,
        )

        # 60% compression ratio gives enough budget for keyword lines
        result = compress_chunk(chunk, ratio=0.6, summarizer=None, query="cosine semantic")

        # Should preserve keyword-matching lines
        assert "cosine" in result
        assert "semantic" in result

    def test_falls_back_to_head_tail_without_query(self):
        """Without query, should use head+tail truncation."""
        chunk = Chunk(
            text="HEAD" + "x" * 500 + "TAIL",
            start_idx=0,
            end_idx=1,
            chunk_type="code",
            relevance_score=0.5,
        )

        result = compress_chunk(chunk, ratio=0.5, summarizer=None, query="")

        # Should preserve head and tail
        assert result.startswith("HEAD")
        assert result.endswith("TAIL") or "snip" in result.lower()

    def test_no_compression_at_high_ratio(self):
        """ratio >= 0.95 should return original text unchanged."""
        original = "def foo(): pass"
        chunk = Chunk(
            text=original,
            start_idx=0,
            end_idx=1,
            chunk_type="code",
            relevance_score=1.0,
        )

        result = compress_chunk(chunk, ratio=0.98, summarizer=None, query="foo")
        assert result == original
