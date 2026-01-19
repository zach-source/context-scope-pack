"""Performance benchmarks - latency by content size."""

import time
from collections import defaultdict

import numpy as np

from benchmarks.config import BenchmarkConfig, BenchmarkResult
from benchmarks.corpus import CorpusFile
from benchmarks.mock_embedder import create_embedder
from scopepack.scope import Embedder, compress_with_scope_indexed


class PerformanceRunner:
    """Benchmark latency by content size."""

    def __init__(self, config: BenchmarkConfig, embedder: Embedder | None = None):
        """Initialize performance runner.

        Args:
            config: Benchmark configuration
            embedder: Embedder to use. Creates one from config if None.
        """
        self.config = config
        self.embedder = embedder or create_embedder(config.use_real_models)

    def run(self, files: list[CorpusFile]) -> list[BenchmarkResult]:
        """Run performance benchmarks with multiple iterations.

        Tests compression latency across different file sizes.

        Args:
            files: List of corpus files to benchmark

        Returns:
            List of benchmark results
        """
        results: list[BenchmarkResult] = []

        for file in files:
            # Run multiple iterations for statistical significance
            for iteration in range(self.config.iterations):
                result = self._run_iteration(file, iteration)
                results.append(result)

        return results

    def _run_iteration(self, file: CorpusFile, iteration: int) -> BenchmarkResult:
        """Run a single performance iteration.

        Args:
            file: File to compress
            iteration: Iteration number

        Returns:
            BenchmarkResult with timing
        """
        budget = 900  # Standard budget for performance tests
        original_tokens = file.estimated_tokens

        # Time the compression
        start = time.perf_counter()
        compressed, symbol_index = compress_with_scope_indexed(
            text=file.content,
            query="",
            budget_tokens=budget,
            embedder=self.embedder,
            summarizer=None,
            file_type=file.file_type,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        compressed_tokens = len(compressed) // 4

        return BenchmarkResult(
            scenario_name=f"performance_{file.size_category}_iter{iteration}",
            file_path=str(file.path),
            file_type=file.file_type,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            budget_tokens=budget,
            budget_adherence=compressed_tokens / budget if budget > 0 else 0.0,
            elapsed_ms=elapsed_ms,
            query=None,
            method="scope",
            extra_data={"size_category": file.size_category, "iteration": iteration},
        )

    def get_latency_percentiles(self, results: list[BenchmarkResult]) -> dict:
        """Calculate latency percentiles by size category.

        Args:
            results: List of performance benchmark results

        Returns:
            Dictionary with percentile latencies by size category
        """
        # Group results by size category
        by_category: dict[str, list[float]] = defaultdict(list)

        for r in results:
            if r.extra_data and "size_category" in r.extra_data:
                category = r.extra_data["size_category"]
                by_category[category].append(r.elapsed_ms)

        # Calculate percentiles for each category
        percentiles = {}
        for category, latencies in by_category.items():
            if latencies:
                latencies_arr = np.array(latencies)
                percentiles[category] = {
                    "count": len(latencies),
                    "min_ms": float(np.min(latencies_arr)),
                    "p50_ms": float(np.percentile(latencies_arr, 50)),
                    "p95_ms": float(np.percentile(latencies_arr, 95)),
                    "p99_ms": float(np.percentile(latencies_arr, 99)),
                    "max_ms": float(np.max(latencies_arr)),
                    "mean_ms": float(np.mean(latencies_arr)),
                    "std_ms": float(np.std(latencies_arr)),
                }

        # Calculate overall percentiles
        all_latencies = [r.elapsed_ms for r in results]
        if all_latencies:
            all_arr = np.array(all_latencies)
            percentiles["overall"] = {
                "count": len(all_latencies),
                "min_ms": float(np.min(all_arr)),
                "p50_ms": float(np.percentile(all_arr, 50)),
                "p95_ms": float(np.percentile(all_arr, 95)),
                "p99_ms": float(np.percentile(all_arr, 99)),
                "max_ms": float(np.max(all_arr)),
                "mean_ms": float(np.mean(all_arr)),
                "std_ms": float(np.std(all_arr)),
            }

        return percentiles

    def check_targets(self, results: list[BenchmarkResult]) -> dict:
        """Check if latency targets are met.

        Targets:
        - tiny: <10ms p95
        - small: <20ms p95
        - medium: <50ms p95
        - large: <100ms p95

        Args:
            results: List of performance benchmark results

        Returns:
            Dictionary with target compliance
        """
        targets = {
            "tiny": 10.0,
            "small": 20.0,
            "medium": 50.0,
            "large": 100.0,
        }

        percentiles = self.get_latency_percentiles(results)

        compliance = {}
        for category, target_ms in targets.items():
            if category in percentiles:
                actual_p95 = percentiles[category]["p95_ms"]
                compliance[category] = {
                    "target_p95_ms": target_ms,
                    "actual_p95_ms": actual_p95,
                    "meets_target": actual_p95 <= target_ms,
                    "margin_ms": target_ms - actual_p95,
                }

        return compliance
