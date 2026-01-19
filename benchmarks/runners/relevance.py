"""Relevance scoring quality benchmarks."""

import time

import numpy as np

from benchmarks.config import BenchmarkConfig, BenchmarkResult, RelevanceScenario
from benchmarks.corpus import CorpusFile
from benchmarks.mock_embedder import create_embedder
from scopepack.scope import Embedder, chunk_text, score_relevance

# Predefined scenarios for testing relevance scoring
# Each scenario specifies which file(s) it should test against
RELEVANCE_SCENARIOS = [
    RelevanceScenario(
        name="compress_function_scope",
        query="compress chunk text",
        expected_high_relevance=["def compress_chunk", "compress_with_scope", "compressed_text"],
        expected_low_relevance=["import re", "Protocol", "detect_file_type"],
    ),
    RelevanceScenario(
        name="embedding_similarity",
        query="embedding vector cosine similarity score",
        expected_high_relevance=["score_relevance", "cosine", "np.dot", "embeddings"],
        expected_low_relevance=["chunk_prose", "HEADING_PATTERN", "BLANK_LINE"],
    ),
    RelevanceScenario(
        name="cache_database",
        query="database cache store retrieve",
        expected_high_relevance=["CacheDB", "get_compressed", "put_compressed", "aiosqlite"],
        expected_low_relevance=["def main", "@app.get"],
    ),
    RelevanceScenario(
        name="http_api_endpoint",
        query="HTTP API endpoint request response",
        expected_high_relevance=[
            "@app.post",
            "CompressRequest",
            "CompressResponse",
            "async def compress",
        ],
        expected_low_relevance=["Protocol", "dataclass", "BLANK_LINE_PATTERN"],
    ),
    RelevanceScenario(
        name="symbol_index_navigation",
        query="symbol index line number navigation",
        expected_high_relevance=["build_symbol_index", "extract_symbol_name", "line_ref"],
        expected_low_relevance=["import", "from dataclasses"],
    ),
]


class RelevanceRunner:
    """Benchmark relevance scoring quality."""

    def __init__(self, config: BenchmarkConfig, embedder: Embedder | None = None):
        """Initialize relevance runner.

        Args:
            config: Benchmark configuration
            embedder: Embedder to use. Creates one from config if None.
        """
        self.config = config
        self.embedder = embedder or create_embedder(config.use_real_models)

    def run(self, files: list[CorpusFile]) -> list[BenchmarkResult]:
        """Run relevance scoring benchmarks.

        Tests predefined scenarios with expected high/low relevance chunks.
        Only tests files that contain the expected patterns.

        Args:
            files: List of corpus files to benchmark

        Returns:
            List of benchmark results
        """
        results: list[BenchmarkResult] = []

        # Only test code files for relevance scenarios
        code_files = [f for f in files if f.file_type == "code"]

        for file in code_files:
            for scenario in RELEVANCE_SCENARIOS:
                # Only run scenario if file contains at least one high-relevance pattern
                has_high_relevance = any(
                    pattern.lower() in file.content.lower()
                    for pattern in scenario.expected_high_relevance
                )
                if not has_high_relevance:
                    continue

                result = self._run_relevance_scenario(file, scenario)
                if result:
                    results.append(result)

        return results

    def _run_relevance_scenario(
        self,
        file: CorpusFile,
        scenario: RelevanceScenario,
    ) -> BenchmarkResult | None:
        """Run a single relevance scoring scenario.

        Args:
            file: File to analyze
            scenario: Relevance scenario to test

        Returns:
            BenchmarkResult with relevance metrics, or None if file not suitable
        """
        # Chunk the file
        chunks = chunk_text(file.content, file.file_type)
        if not chunks:
            return None

        # Time the relevance scoring
        start = time.perf_counter()
        scored_chunks = score_relevance(chunks, scenario.query, self.embedder)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Analyze scores
        scores = [c.relevance_score for c in scored_chunks]
        if not scores:
            return None

        # Categorize chunks by expected relevance
        high_scores: list[float] = []
        low_scores: list[float] = []
        high_chunks: list[str] = []
        low_chunks: list[str] = []

        for chunk in scored_chunks:
            chunk_text_lower = chunk.text.lower()

            # Check if chunk matches expected high relevance
            matched_high = False
            for expected in scenario.expected_high_relevance:
                if expected.lower() in chunk_text_lower:
                    high_scores.append(chunk.relevance_score)
                    high_chunks.append(expected)
                    matched_high = True
                    break

            if not matched_high:
                # Check if chunk matches expected low relevance
                for expected in scenario.expected_low_relevance:
                    if expected.lower() in chunk_text_lower:
                        low_scores.append(chunk.relevance_score)
                        low_chunks.append(expected)
                        break

        # Calculate separation metric
        high_vs_low_separation = 0.0
        if high_scores and low_scores:
            high_vs_low_separation = float(np.mean(high_scores) - np.mean(low_scores))

        # Score distribution statistics
        relevance_data = {
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "high_relevance_mean": float(np.mean(high_scores)) if high_scores else None,
            "low_relevance_mean": float(np.mean(low_scores)) if low_scores else None,
            "high_vs_low_separation": high_vs_low_separation,
            "high_count": len(high_scores),
            "low_count": len(low_scores),
            "high_patterns_matched": high_chunks,
            "low_patterns_matched": low_chunks,
        }

        return BenchmarkResult(
            scenario_name=f"relevance_{scenario.name}",
            file_path=str(file.path),
            file_type=file.file_type,
            original_tokens=file.estimated_tokens,
            compressed_tokens=0,  # Not applicable for relevance benchmark
            compression_ratio=0.0,
            budget_tokens=0,
            budget_adherence=0.0,
            elapsed_ms=elapsed_ms,
            query=scenario.query,
            method="scope",
            symbol_count=len(chunks),
            relevance_scores=relevance_data,
            extra_data={"separation_target": 0.1},  # Realistic target based on observations
        )

    def get_summary_metrics(self, results: list[BenchmarkResult]) -> dict:
        """Calculate summary metrics for relevance benchmarks.

        Args:
            results: List of relevance benchmark results

        Returns:
            Dictionary with summary statistics
        """
        separations: list[float] = []

        for r in results:
            if r.relevance_scores and r.relevance_scores.get("high_vs_low_separation"):
                separations.append(r.relevance_scores["high_vs_low_separation"])

        if not separations:
            return {"avg_separation": 0.0, "separations_above_target": 0}

        target = 0.1  # Realistic target based on real embeddings
        return {
            "avg_separation": float(np.mean(separations)),
            "max_separation": float(np.max(separations)),
            "min_separation": float(np.min(separations)),
            "separations_above_target": sum(1 for s in separations if s >= target),
            "total_scenarios": len(separations),
            "target": target,
        }
