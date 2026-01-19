"""Benchmark SCOPE vs quick_compress fallback comparison."""

import time

from benchmarks.config import BenchmarkConfig, BenchmarkResult
from benchmarks.corpus import CorpusFile
from benchmarks.mock_embedder import create_embedder
from scopepack.scope import (
    Embedder,
    compress_with_scope_indexed,
    quick_compress_indexed,
)


class FallbackRunner:
    """Compare SCOPE compression with quick_compress fallback."""

    def __init__(self, config: BenchmarkConfig, embedder: Embedder | None = None):
        """Initialize fallback comparison runner.

        Args:
            config: Benchmark configuration
            embedder: Embedder to use. Creates one from config if None.
        """
        self.config = config
        self.embedder = embedder or create_embedder(config.use_real_models)

    def run(self, files: list[CorpusFile]) -> list[BenchmarkResult]:
        """Run SCOPE vs quick_compress comparison.

        Tests the same inputs with both methods to compare:
        - Compression quality (ratio, symbol preservation)
        - Performance (latency)

        Args:
            files: List of corpus files to benchmark

        Returns:
            List of benchmark results (paired SCOPE/quick results)
        """
        results: list[BenchmarkResult] = []

        for file in files:
            for budget in self.config.budget_tokens:
                # Test with no query (baseline comparison)
                scope_result, quick_result = self._run_comparison(
                    file=file,
                    budget=budget,
                    query=None,
                )
                results.extend([scope_result, quick_result])

                # Test with a query (relevance-aware comparison)
                query = self._extract_query(file)
                if query:
                    scope_result, quick_result = self._run_comparison(
                        file=file,
                        budget=budget,
                        query=query,
                    )
                    results.extend([scope_result, quick_result])

        return results

    def _run_comparison(
        self,
        file: CorpusFile,
        budget: int,
        query: str | None,
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """Run both SCOPE and quick_compress on same input.

        Args:
            file: File to compress
            budget: Token budget
            query: Optional query for relevance scoring

        Returns:
            Tuple of (scope_result, quick_result)
        """
        original_tokens = file.estimated_tokens
        query_suffix = "with_query" if query else "no_query"

        # Run SCOPE compression
        start = time.perf_counter()
        scope_compressed, scope_index = compress_with_scope_indexed(
            text=file.content,
            query=query or "",
            budget_tokens=budget,
            embedder=self.embedder,
            summarizer=None,
            file_type=file.file_type,
        )
        scope_elapsed = (time.perf_counter() - start) * 1000

        scope_tokens = len(scope_compressed) // 4
        scope_symbols = scope_index.count("- L") if scope_index else 0

        scope_result = BenchmarkResult(
            scenario_name=f"fallback_scope_{budget}_{query_suffix}",
            file_path=str(file.path),
            file_type=file.file_type,
            original_tokens=original_tokens,
            compressed_tokens=scope_tokens,
            compression_ratio=scope_tokens / original_tokens if original_tokens > 0 else 1.0,
            budget_tokens=budget,
            budget_adherence=scope_tokens / budget if budget > 0 else 0.0,
            elapsed_ms=scope_elapsed,
            query=query,
            method="scope",
            symbol_count=scope_symbols,
        )

        # Run quick_compress
        budget_chars = budget * 4
        start = time.perf_counter()
        quick_compressed, quick_index = quick_compress_indexed(
            text=file.content,
            budget_chars=budget_chars,
            file_type=file.file_type,
        )
        quick_elapsed = (time.perf_counter() - start) * 1000

        quick_tokens = len(quick_compressed) // 4
        quick_symbols = quick_index.count("- L") if quick_index else 0

        quick_result = BenchmarkResult(
            scenario_name=f"fallback_quick_{budget}_{query_suffix}",
            file_path=str(file.path),
            file_type=file.file_type,
            original_tokens=original_tokens,
            compressed_tokens=quick_tokens,
            compression_ratio=quick_tokens / original_tokens if original_tokens > 0 else 1.0,
            budget_tokens=budget,
            budget_adherence=quick_tokens / budget if budget > 0 else 0.0,
            elapsed_ms=quick_elapsed,
            query=query,
            method="quick",
            symbol_count=quick_symbols,
        )

        return scope_result, quick_result

    def _extract_query(self, file: CorpusFile) -> str | None:
        """Extract a reasonable query from the file for testing.

        Returns the first meaningful line (function def, class, etc.)
        """
        for line in file.content.split("\n")[:50]:
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "async def ")):
                # Extract the function/class name
                parts = stripped.split("(")[0].split()
                if len(parts) >= 2:
                    return parts[-1]  # Return the name
        return None

    def get_comparison_summary(self, results: list[BenchmarkResult]) -> dict:
        """Calculate comparison summary between SCOPE and quick.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary with comparison metrics
        """
        scope_results = [r for r in results if r.method == "scope"]
        quick_results = [r for r in results if r.method == "quick"]

        if not scope_results or not quick_results:
            return {"comparison_available": False}

        # Pair up results by file and budget
        scope_better = 0
        quick_better = 0
        scope_faster = 0
        quick_faster = 0

        for scope_r in scope_results:
            # Find matching quick result
            for quick_r in quick_results:
                if (
                    scope_r.file_path == quick_r.file_path
                    and scope_r.budget_tokens == quick_r.budget_tokens
                    and scope_r.query == quick_r.query
                ):
                    # Compare compression quality
                    # Lower compression_ratio = better (more compression)
                    if scope_r.compression_ratio < quick_r.compression_ratio:
                        scope_better += 1
                    else:
                        quick_better += 1

                    # Compare latency
                    if scope_r.elapsed_ms < quick_r.elapsed_ms:
                        scope_faster += 1
                    else:
                        quick_faster += 1
                    break

        total_comparisons = scope_better + quick_better
        return {
            "comparison_available": True,
            "scope_better_compression": scope_better,
            "quick_better_compression": quick_better,
            "scope_compression_win_rate": scope_better / total_comparisons
            if total_comparisons
            else 0,
            "scope_faster": scope_faster,
            "quick_faster": quick_faster,
            "scope_speed_win_rate": scope_faster / (scope_faster + quick_faster)
            if (scope_faster + quick_faster)
            else 0,
        }
