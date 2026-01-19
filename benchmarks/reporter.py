"""Benchmark result reporting - JSON, CSV, and console output."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfig, BenchmarkResult, BenchmarkSummary


class BenchmarkReporter:
    """Generate reports from benchmark results."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize reporter.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.output_dir = config.output_dir

    def generate_summary(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Generate summary statistics from results.

        Args:
            results: List of all benchmark results

        Returns:
            BenchmarkSummary with aggregated metrics
        """
        if not results:
            return BenchmarkSummary(
                total_files=0,
                total_scenarios=0,
                avg_compression_ratio=0.0,
                avg_budget_adherence=0.0,
                avg_elapsed_ms=0.0,
                p50_elapsed_ms=0.0,
                p95_elapsed_ms=0.0,
                p99_elapsed_ms=0.0,
                scope_better_count=0,
                scope_worse_count=0,
            )

        # Unique files
        unique_files = {r.file_path for r in results}

        # Filter for results with compression data
        compression_results = [r for r in results if r.compression_ratio > 0]

        # Calculate compression stats
        ratios = [r.compression_ratio for r in compression_results] if compression_results else [0]
        adherences = (
            [r.budget_adherence for r in compression_results] if compression_results else [0]
        )

        # Calculate latency percentiles
        latencies = [r.elapsed_ms for r in results]
        latencies_arr = np.array(latencies)

        # Count SCOPE vs quick wins
        scope_results = [r for r in results if r.method == "scope"]
        quick_results = [r for r in results if r.method == "quick"]

        scope_better = 0
        scope_worse = 0

        for scope_r in scope_results:
            for quick_r in quick_results:
                if (
                    scope_r.file_path == quick_r.file_path
                    and scope_r.budget_tokens == quick_r.budget_tokens
                    and "fallback" in scope_r.scenario_name
                    and "fallback" in quick_r.scenario_name
                ):
                    if scope_r.compression_ratio < quick_r.compression_ratio:
                        scope_better += 1
                    elif scope_r.compression_ratio > quick_r.compression_ratio:
                        scope_worse += 1
                    break

        return BenchmarkSummary(
            total_files=len(unique_files),
            total_scenarios=len(results),
            avg_compression_ratio=float(np.mean(ratios)),
            avg_budget_adherence=float(np.mean(adherences)),
            avg_elapsed_ms=float(np.mean(latencies)),
            p50_elapsed_ms=float(np.percentile(latencies_arr, 50)),
            p95_elapsed_ms=float(np.percentile(latencies_arr, 95)),
            p99_elapsed_ms=float(np.percentile(latencies_arr, 99)),
            scope_better_count=scope_better,
            scope_worse_count=scope_worse,
        )

    def write_json(
        self,
        results: list[BenchmarkResult],
        summary: BenchmarkSummary,
        extra_data: dict[str, Any] | None = None,
        filename: str = "results.json",
    ) -> Path:
        """Write results to JSON file.

        Args:
            results: List of benchmark results
            summary: Summary statistics
            extra_data: Additional data to include
            filename: Output filename

        Returns:
            Path to written file
        """
        output = {
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "config": {
                    "corpus_dir": str(self.config.corpus_dir),
                    "budget_tokens": self.config.budget_tokens,
                    "iterations": self.config.iterations,
                    "use_real_models": self.config.use_real_models,
                },
            },
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        if extra_data:
            output["extra"] = extra_data

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return output_path

    def write_csv(
        self,
        results: list[BenchmarkResult],
        filename: str = "results.csv",
    ) -> Path:
        """Write results to CSV file.

        Args:
            results: List of benchmark results
            filename: Output filename

        Returns:
            Path to written file
        """
        if not results:
            return self.output_dir / filename

        output_path = self.output_dir / filename

        fieldnames = [
            "scenario_name",
            "file_path",
            "file_type",
            "original_tokens",
            "compressed_tokens",
            "compression_ratio",
            "budget_tokens",
            "budget_adherence",
            "elapsed_ms",
            "query",
            "method",
            "symbol_count",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_csv_row())

        return output_path

    def print_console_summary(
        self,
        summary: BenchmarkSummary,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        """Print summary to console.

        Args:
            summary: Summary statistics
            extra_data: Additional metrics to display
        """
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"\nTotal files tested: {summary.total_files}")
        print(f"Total scenarios run: {summary.total_scenarios}")

        print("\n--- Compression Metrics ---")
        print(f"Avg compression ratio: {summary.avg_compression_ratio:.3f}")
        print(f"Avg budget adherence: {summary.avg_budget_adherence:.3f}")

        print("\n--- Latency Metrics ---")
        print(f"Avg latency: {summary.avg_elapsed_ms:.2f}ms")
        print(f"P50 latency: {summary.p50_elapsed_ms:.2f}ms")
        print(f"P95 latency: {summary.p95_elapsed_ms:.2f}ms")
        print(f"P99 latency: {summary.p99_elapsed_ms:.2f}ms")

        print("\n--- SCOPE vs Quick Comparison ---")
        print(f"SCOPE better: {summary.scope_better_count}")
        print(f"Quick better: {summary.scope_worse_count}")

        if extra_data:
            print("\n--- Additional Metrics ---")
            for key, value in extra_data.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"  {k}: {v:.3f}")
                        else:
                            print(f"  {k}: {v}")
                elif isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")

        print("\n" + "=" * 60)

    def print_target_compliance(self, compliance: dict[str, Any]) -> None:
        """Print latency target compliance.

        Args:
            compliance: Dictionary from PerformanceRunner.check_targets
        """
        print("\n--- Latency Target Compliance ---")
        for category, data in compliance.items():
            status = "✓" if data["meets_target"] else "✗"
            print(
                f"  {category}: {status} "
                f"(target: {data['target_p95_ms']}ms, "
                f"actual: {data['actual_p95_ms']:.2f}ms, "
                f"margin: {data['margin_ms']:.2f}ms)"
            )
