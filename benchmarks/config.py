"""Configuration and result types for benchmarks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    corpus_dir: Path
    output_dir: Path
    budget_tokens: list[int] = field(default_factory=lambda: [100, 500, 900])
    iterations: int = 3
    use_real_models: bool = False

    def __post_init__(self) -> None:
        """Ensure paths are Path objects and directories exist."""
        self.corpus_dir = Path(self.corpus_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark scenario."""

    scenario_name: str
    file_path: str
    file_type: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    budget_tokens: int
    budget_adherence: float  # compressed/budget - ideally 0.8-1.0
    elapsed_ms: float
    query: str | None
    method: str  # "scope", "quick"

    # Optional fields for specific benchmarks
    symbol_count: int | None = None
    relevance_scores: dict[str, float] | None = None
    extra_data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "budget_tokens": self.budget_tokens,
            "budget_adherence": self.budget_adherence,
            "elapsed_ms": self.elapsed_ms,
            "query": self.query,
            "method": self.method,
            "symbol_count": self.symbol_count,
            "relevance_scores": self.relevance_scores,
            "extra_data": self.extra_data,
        }

    def to_csv_row(self) -> dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        return {
            "scenario_name": self.scenario_name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "budget_tokens": self.budget_tokens,
            "budget_adherence": self.budget_adherence,
            "elapsed_ms": self.elapsed_ms,
            "query": self.query or "",
            "method": self.method,
            "symbol_count": self.symbol_count or 0,
        }


@dataclass
class RelevanceScenario:
    """A predefined scenario for testing relevance scoring."""

    name: str
    query: str
    expected_high_relevance: list[str]  # Substrings that should score high
    expected_low_relevance: list[str]  # Substrings that should score low


@dataclass
class BenchmarkSummary:
    """Summary statistics across all benchmark results."""

    total_files: int
    total_scenarios: int
    avg_compression_ratio: float
    avg_budget_adherence: float
    avg_elapsed_ms: float
    p50_elapsed_ms: float
    p95_elapsed_ms: float
    p99_elapsed_ms: float
    scope_better_count: int  # Times SCOPE beat fallback
    scope_worse_count: int  # Times fallback beat SCOPE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "total_scenarios": self.total_scenarios,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_budget_adherence": self.avg_budget_adherence,
            "avg_elapsed_ms": self.avg_elapsed_ms,
            "p50_elapsed_ms": self.p50_elapsed_ms,
            "p95_elapsed_ms": self.p95_elapsed_ms,
            "p99_elapsed_ms": self.p99_elapsed_ms,
            "scope_better_count": self.scope_better_count,
            "scope_worse_count": self.scope_worse_count,
        }
