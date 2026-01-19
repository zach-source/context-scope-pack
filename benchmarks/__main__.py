"""CLI entry point for benchmark suite."""

import argparse
import sys
from pathlib import Path

from .config import BenchmarkConfig
from .corpus import create_default_corpus
from .llm_evaluator import create_evaluator
from .mock_embedder import create_embedder
from .reporter import BenchmarkReporter
from .runners import (
    CompressionRunner,
    FallbackRunner,
    MCPRunner,
    PerformanceRunner,
    QualityRunner,
    RelevanceRunner,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ScopePack Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks                          # Run all benchmarks
  python -m benchmarks --budgets 100 500 1500   # Custom token budgets
  python -m benchmarks --runners compression    # Run only compression benchmarks
  python -m benchmarks --runners quality --llm-eval --aws-profile myprofile  # Live LLM eval
        """,
    )

    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Project root directory for corpus discovery (auto-detects if not set)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for output files (default: benchmark_results/)",
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[100, 500, 900],
        help="Token budgets to test (default: 100 500 900)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for performance benchmarks (default: 3)",
    )

    # --real-models flag removed: benchmarks always use real models for meaningful results

    parser.add_argument(
        "--runners",
        nargs="+",
        choices=[
            "compression",
            "relevance",
            "fallback",
            "performance",
            "quality",
            "mcp",
            "all",
        ],
        default=["all"],
        help="Which benchmark runners to execute (default: all)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (still writes files)",
    )

    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Use live LLM evaluation via AWS Bedrock (requires --aws-profile)",
    )

    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile to assume for Bedrock access (uses 'assume' CLI)",
    )

    parser.add_argument(
        "--aws-region",
        type=str,
        default="us-west-2",
        help="AWS region for Bedrock (default: us-west-2)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine corpus directory
    corpus_dir = args.corpus_dir
    if corpus_dir is None:
        # Auto-detect project root
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                corpus_dir = current
                break
            current = current.parent
        else:
            corpus_dir = Path.cwd()

    # Create configuration
    config = BenchmarkConfig(
        corpus_dir=corpus_dir,
        output_dir=args.output_dir,
        budget_tokens=args.budgets,
        iterations=args.iterations,
        use_real_models=True,  # Always use real models
    )

    if not args.quiet:
        print("ScopePack Benchmark Suite")
        print(f"  Corpus: {config.corpus_dir}")
        print(f"  Output: {config.output_dir}")
        print(f"  Budgets: {config.budget_tokens}")
        print()

    # Discover corpus
    corpus = create_default_corpus(config.corpus_dir)
    files = corpus.discover()

    if not files:
        print("ERROR: No files found in corpus", file=sys.stderr)
        return 1

    if not args.quiet:
        summary = corpus.get_summary()
        print(
            f"Corpus: {summary['total_files']} files, {summary['total_estimated_tokens']} estimated tokens"
        )
        print(f"  By type: {summary['by_type']}")
        print(f"  By size: {summary['by_size_category']}")
        print()

    # Create shared embedder
    embedder = create_embedder(config.use_real_models)

    # Determine which runners to execute
    runners_to_run = set(args.runners)
    if "all" in runners_to_run:
        runners_to_run = {
            "compression",
            "relevance",
            "fallback",
            "performance",
            "quality",
        }

    # Run benchmarks
    all_results = []
    extra_data = {}

    if "compression" in runners_to_run:
        if not args.quiet:
            print("Running compression benchmarks...")
        runner = CompressionRunner(config, embedder)
        results = runner.run(files)
        all_results.extend(results)
        if not args.quiet:
            print(f"  {len(results)} scenarios completed")

    if "relevance" in runners_to_run:
        if not args.quiet:
            print("Running relevance benchmarks...")
        runner = RelevanceRunner(config, embedder)
        results = runner.run(files)
        all_results.extend(results)
        extra_data["relevance_metrics"] = runner.get_summary_metrics(results)
        if not args.quiet:
            print(f"  {len(results)} scenarios completed")

    if "fallback" in runners_to_run:
        if not args.quiet:
            print("Running fallback comparison benchmarks...")
        runner = FallbackRunner(config, embedder)
        results = runner.run(files)
        all_results.extend(results)
        extra_data["fallback_comparison"] = runner.get_comparison_summary(results)
        if not args.quiet:
            print(f"  {len(results)} scenarios completed")

    if "performance" in runners_to_run:
        if not args.quiet:
            print("Running performance benchmarks...")
        runner = PerformanceRunner(config, embedder)
        results = runner.run(files)
        all_results.extend(results)
        extra_data["latency_percentiles"] = runner.get_latency_percentiles(results)
        extra_data["target_compliance"] = runner.check_targets(results)
        if not args.quiet:
            print(f"  {len(results)} scenarios completed")

    if "quality" in runners_to_run:
        llm_evaluator = None
        if args.llm_eval:
            if not args.quiet:
                print("Running LLM quality benchmarks with LIVE Bedrock evaluation...")
            llm_evaluator = create_evaluator(
                use_llm=True,
                aws_profile=args.aws_profile,
                aws_region=args.aws_region,
            )
            if llm_evaluator is None:
                print("  WARNING: LLM evaluation requested but failed to initialize")
                print("  Falling back to string matching")
        else:
            if not args.quiet:
                print("Running LLM quality benchmarks (string matching)...")

        runner = QualityRunner(config, embedder, llm_evaluator=llm_evaluator)
        results = runner.run(files)
        all_results.extend(results)
        extra_data["quality_comparison"] = runner.get_quality_summary(results)
        if not args.quiet:
            print(f"  {len(results)} scenarios completed")

    if "mcp" in runners_to_run:
        llm_evaluator = None
        if args.llm_eval:
            if not args.quiet:
                print("Running MCP compression benchmarks with LIVE Context7 + LLM evaluation...")
            llm_evaluator = create_evaluator(
                use_llm=True,
                aws_profile=args.aws_profile,
                aws_region=args.aws_region,
            )
            if llm_evaluator is None:
                print("  WARNING: LLM evaluation required for MCP benchmarks")
                print("  Skipping MCP benchmarks (use --llm-eval --aws-profile)")
        else:
            if not args.quiet:
                print("  WARNING: MCP benchmarks require --llm-eval flag")
                print("  Skipping MCP benchmarks")

        if llm_evaluator:
            runner = MCPRunner(config, embedder, llm_evaluator=llm_evaluator)
            results = runner.run(files)
            all_results.extend(results)
            extra_data["mcp_comparison"] = runner.get_mcp_summary(results)
            if not args.quiet:
                print(f"  {len(results)} scenarios completed")

    # Generate reports
    reporter = BenchmarkReporter(config)
    summary = reporter.generate_summary(all_results)

    json_path = reporter.write_json(all_results, summary, extra_data)
    csv_path = reporter.write_csv(all_results)

    if not args.quiet:
        print()
        reporter.print_console_summary(summary, extra_data)
        if "target_compliance" in extra_data:
            reporter.print_target_compliance(extra_data["target_compliance"])
        print("\nResults written to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
