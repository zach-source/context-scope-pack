#!/usr/bin/env python
"""Compare embedding models on code-specific quality benchmarks.

Usage:
    python -m benchmarks.compare_embedders [--models MODEL1,MODEL2,...]

Requires AWS credentials for Bedrock models.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# Models to compare
DEFAULT_MODELS = [
    "bge-small-en-v1.5",  # Local baseline
    "cohere-embed-v4:1024",  # Bedrock Cohere v4
]

# Code-specific test cases (subset of quality runner)
CODE_TEST_CASES = [
    "find_method_chunk_code",
    "find_method_tokenize_bm25",
    "find_variable_bm25_params",
    "find_variable_alpha_temperature",
    "find_function_usage_embedder_encode",
    "find_function_usage_np_dot",
]


def run_benchmark(embedder: str, budget: int = 900) -> dict:
    """Run quality benchmark with specified embedder."""
    env = os.environ.copy()
    env["SCOPE_EMBEDDER"] = embedder

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks",
            "--runners",
            "quality",
            "--budgets",
            str(budget),
            "--output",
            f"benchmark_results/results_{embedder.replace(':', '_')}.json",
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    if result.returncode != 0:
        print(f"Error running benchmark for {embedder}:")
        print(result.stderr)
        return {}

    # Parse results
    csv_path = Path(__file__).parent.parent / "benchmark_results" / "results.csv"
    results = {}
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["method"] == "scope":
                    name = (
                        row["scenario_name"]
                        .replace("quality_", "")
                        .replace(f"_{budget}", "")
                    )
                    results[name] = {
                        "quality": float(row.get("quality", 0)),
                        "compression_ratio": float(row["compression_ratio"]),
                        "elapsed_ms": float(row["elapsed_ms"]),
                    }
    return results


def compare_models(models: list[str], budget: int = 900):
    """Compare multiple embedding models."""
    print(f"\n{'='*60}")
    print(f"Embedding Model Comparison (budget={budget} tokens)")
    print(f"{'='*60}\n")

    all_results = {}
    for model in models:
        print(f"Running benchmark with {model}...")
        all_results[model] = run_benchmark(model, budget)
        print("  Done.\n")

    # Print comparison table
    print(f"\n{'Test Case':<40} ", end="")
    for model in models:
        short_name = model.split("/")[-1][:15]
        print(f"{short_name:>15} ", end="")
    print()
    print("-" * (40 + 16 * len(models)))

    for test in CODE_TEST_CASES:
        print(f"{test:<40} ", end="")
        for model in models:
            if test in all_results.get(model, {}):
                q = all_results[model][test].get("quality", 0)
                print(f"{q:>14.1%} ", end="")
            else:
                print(f"{'N/A':>15} ", end="")
        print()

    # Summary
    print(f"\n{'='*60}")
    print("Summary (Code Tests Only)")
    print(f"{'='*60}")

    for model in models:
        results = all_results.get(model, {})
        code_results = {k: v for k, v in results.items() if k in CODE_TEST_CASES}
        if code_results:
            avg_quality = sum(r.get("quality", 0) for r in code_results.values()) / len(
                code_results
            )
            avg_latency = sum(
                r.get("elapsed_ms", 0) for r in code_results.values()
            ) / len(code_results)
            print(f"{model}:")
            print(f"  Avg Quality: {avg_quality:.1%}")
            print(f"  Avg Latency: {avg_latency:.0f}ms")
            print()


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of models to compare",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=900,
        help="Token budget for compression",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    compare_models(models, args.budget)


if __name__ == "__main__":
    main()
