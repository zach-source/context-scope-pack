"""Compression ratio benchmarks."""

import time

from benchmarks.config import BenchmarkConfig, BenchmarkResult
from benchmarks.corpus import CorpusFile
from benchmarks.mock_embedder import create_embedder
from scopepack.scope import Embedder, compress_with_scope_indexed


class CompressionRunner:
    """Benchmark compression at multiple budget levels."""

    def __init__(self, config: BenchmarkConfig, embedder: Embedder | None = None):
        """Initialize compression runner.

        Args:
            config: Benchmark configuration
            embedder: Embedder to use. Creates one from config if None.
        """
        self.config = config
        self.embedder = embedder or create_embedder(config.use_real_models)

    def run(self, files: list[CorpusFile]) -> list[BenchmarkResult]:
        """Run compression benchmarks on corpus files.

        Tests each file at each budget level with different query scenarios:
        - No query (equal relevance)
        - First-line query (simulating code navigation)
        - Keyword query (simulating search)

        Args:
            files: List of corpus files to benchmark

        Returns:
            List of benchmark results
        """
        results: list[BenchmarkResult] = []

        for file in files:
            for budget in self.config.budget_tokens:
                # Scenario 1: No query
                results.append(
                    self._run_compression(
                        file=file,
                        budget=budget,
                        query=None,
                        scenario_name=f"compression_no_query_{budget}",
                    )
                )

                # Scenario 2: First-line query
                first_line = file.content.split("\n")[0].strip()
                if first_line:
                    results.append(
                        self._run_compression(
                            file=file,
                            budget=budget,
                            query=first_line,
                            scenario_name=f"compression_first_line_{budget}",
                        )
                    )

                # Scenario 3: Keyword queries
                for keyword in self._extract_keywords(file):
                    results.append(
                        self._run_compression(
                            file=file,
                            budget=budget,
                            query=keyword,
                            scenario_name=f"compression_keyword_{keyword}_{budget}",
                        )
                    )

        return results

    def _run_compression(
        self,
        file: CorpusFile,
        budget: int,
        query: str | None,
        scenario_name: str,
    ) -> BenchmarkResult:
        """Run a single compression scenario.

        Args:
            file: File to compress
            budget: Token budget
            query: Optional query for relevance scoring
            scenario_name: Name for this scenario

        Returns:
            BenchmarkResult with compression metrics
        """
        original_tokens = file.estimated_tokens

        # Time the compression
        start = time.perf_counter()
        compressed, symbol_index = compress_with_scope_indexed(
            text=file.content,
            query=query or "",
            budget_tokens=budget,
            embedder=self.embedder,
            summarizer=None,  # Use fallback truncation
            file_type=file.file_type,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        compressed_tokens = len(compressed) // 4
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        budget_adherence = compressed_tokens / budget if budget > 0 else 0.0

        # Count symbols in index
        symbol_count = symbol_index.count("- L") if symbol_index else 0

        return BenchmarkResult(
            scenario_name=scenario_name,
            file_path=str(file.path),
            file_type=file.file_type,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            budget_tokens=budget,
            budget_adherence=budget_adherence,
            elapsed_ms=elapsed_ms,
            query=query,
            method="scope",
            symbol_count=symbol_count,
        )

    def _extract_keywords(self, file: CorpusFile) -> list[str]:
        """Extract relevant keywords from file for query testing.

        Returns a small set of keywords that should have high relevance
        in the file.
        """
        keywords: list[str] = []

        # Look for common code patterns
        content = file.content.lower()

        # Check for compression-related keywords (relevant to this project)
        compression_keywords = ["compress", "chunk", "relevance", "embed", "cache"]
        for kw in compression_keywords:
            if kw in content:
                keywords.append(kw)
                if len(keywords) >= 2:
                    break

        return keywords
