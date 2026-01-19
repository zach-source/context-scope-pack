"""LLM quality benchmarks - measure if SCOPE improves LLM task performance.

Tests whether compressed content preserves the information needed for LLMs
to correctly answer questions about the code.

Supports two evaluation modes:
1. String matching (fast, deterministic) - checks if required content is preserved
2. LLM evaluation (live, realistic) - actually asks an LLM to answer questions
"""

import time
from dataclasses import dataclass, field

from benchmarks.config import BenchmarkConfig, BenchmarkResult
from benchmarks.corpus import CorpusFile
from benchmarks.llm_evaluator import LLMEvaluator
from benchmarks.mock_embedder import create_embedder
from scopepack.scope import (
    Embedder,
    compress_with_scope_indexed,
    quick_compress_indexed,
)


@dataclass
class QualityTestCase:
    """A test case for measuring LLM quality on compressed content."""

    name: str
    file_pattern: str  # Which file(s) this test applies to
    query: str  # Query for SCOPE relevance scoring
    questions: list[str]  # Questions an LLM should be able to answer
    expected_answers: list[str]  # Expected answers for LLM judging
    required_content: list[str]  # Content that MUST be preserved to answer questions
    nice_to_have: list[str] = field(default_factory=list)  # Content that helps but isn't critical


# Test cases that verify key information is preserved
QUALITY_TEST_CASES = [
    QualityTestCase(
        name="compression_api",
        file_pattern="scope.py",
        query="compress text with SCOPE algorithm",
        questions=[
            "What is the main function to compress text?",
            "What parameters does compress_with_scope take?",
            "How does the compression ratio get computed?",
        ],
        expected_answers=[
            "compress_with_scope or compress_with_scope_indexed is the main compression function",
            "It takes text, query, budget_tokens, embedder, summarizer, and file_type parameters",
            "Compression ratio is computed as compressed_tokens / original_tokens",
        ],
        required_content=[
            "def compress_with_scope",
            "budget_tokens",
            "embedder",
            "compression_ratio",
        ],
        nice_to_have=[
            "chunk_text",
            "score_relevance",
            "compute_compression_ratios",
        ],
    ),
    QualityTestCase(
        name="chunking_logic",
        file_pattern="scope.py",
        query="chunk code into semantic units functions classes",
        questions=[
            "How does the code chunker detect function boundaries?",
            "What chunk types are supported?",
        ],
        expected_answers=[
            "It uses regex patterns to detect 'def ' and 'class ' keywords at the start of lines",
            "The supported chunk types are 'code' for functions/classes and 'prose' for text/markdown",
        ],
        required_content=[
            "def chunk_code",
            "def chunk_text",
            "chunk_type",
            "def ",
            "class ",
        ],
        nice_to_have=[
            "Chunk",
            "start_idx",
            "end_idx",
        ],
    ),
    QualityTestCase(
        name="relevance_scoring",
        file_pattern="scope.py",
        query="relevance score embedding cosine similarity",
        questions=[
            "How are chunks scored for relevance?",
            "What similarity metric is used?",
        ],
        expected_answers=[
            "Chunks are scored by computing embeddings for both the chunk and query, then measuring similarity",
            "Cosine similarity is used to compare embedding vectors",
        ],
        required_content=[
            "def score_relevance",
            "cosine",
            "embedder.encode",
            "relevance_score",
        ],
        nice_to_have=[
            "np.dot",
            "np.linalg.norm",
        ],
    ),
    QualityTestCase(
        name="http_endpoints",
        file_pattern="daemon.py",
        query="HTTP API endpoints compress request response",
        questions=[
            "What endpoints does the daemon expose?",
            "What is the request format for compression?",
        ],
        expected_answers=[
            "The daemon exposes /compress, /compress/quick, /summarize, /file-summary, and /health endpoints",
            "CompressRequest with fields: content, query, budget_tokens, file_type, and file_path",
        ],
        required_content=[
            "@app.post",
            "/compress",
            "CompressRequest",
            "budget_tokens",
        ],
        nice_to_have=[
            "CompressResponse",
            "compression_ratio",
            "cache_hit",
        ],
    ),
    QualityTestCase(
        name="caching_system",
        file_pattern="db.py",
        query="cache database store retrieve compressed content",
        questions=[
            "How is compressed content cached?",
            "What identifies a cache entry?",
        ],
        expected_answers=[
            "Compressed content is cached in SQLite using aiosqlite with get/put methods",
            "Cache entries are identified by content_hash combined with query and budget parameters",
        ],
        required_content=[
            "class CacheDB",
            "get_compressed",
            "put_compressed",
            "content_hash",
        ],
        nice_to_have=[
            "aiosqlite",
            "model_version",
        ],
    ),
    # Code navigation test cases - find specific code elements
    QualityTestCase(
        name="find_method_chunk_code",
        file_pattern="scope.py",
        query="chunk_code function definition",
        questions=[
            "What does the chunk_code function do?",
            "What does chunk_code return?",
        ],
        expected_answers=[
            "chunk_code chunks code into semantic units like functions and classes",
            "It returns a list of Chunk objects",
        ],
        required_content=[
            "def chunk_code",
            "list[Chunk]",
            "chunks.append",
        ],
        nice_to_have=[
            "in_function_or_class",
            "indent_level",
        ],
    ),
    QualityTestCase(
        name="find_method_tokenize_bm25",
        file_pattern="scope.py",
        query="tokenize_for_bm25 function",
        questions=[
            "How does tokenize_for_bm25 handle camelCase?",
            "What does tokenize_for_bm25 return?",
        ],
        expected_answers=[
            "It uses regex to split camelCase before lowercasing",
            "It returns a list of string tokens",
        ],
        required_content=[
            "def tokenize_for_bm25",
            "camelCase",
            "list[str]",
        ],
        nice_to_have=[
            "snake_case",
            "re.findall",
        ],
    ),
    QualityTestCase(
        name="find_variable_bm25_params",
        file_pattern="scope.py",
        query="BM25 parameters k1 b variables",
        questions=[
            "What are the BM25 parameters?",
            "What values are used for k1 and b?",
        ],
        expected_answers=[
            "BM25 uses k1 and b as parameters",
            "k1=1.5 and b=0.75 are the standard values",
        ],
        required_content=[
            "k1 = 1.5",
            "b = 0.75",
        ],
        nice_to_have=[
            "BM25",
            "tf_component",
            "idf",
        ],
    ),
    QualityTestCase(
        name="find_variable_alpha_temperature",
        file_pattern="scope.py",
        query="alpha temperature hybrid scoring parameters",
        questions=[
            "What does alpha control in hybrid scoring?",
            "What does temperature do?",
        ],
        expected_answers=[
            "Alpha balances between semantic (1.0) and lexical (0.0) scoring",
            "Temperature sharpens score differentiation",
        ],
        required_content=[
            "alpha: float = 0.5",
            "temperature: float = 2.0",
        ],
        nice_to_have=[
            "semantic",
            "lexical",
            "sharpening",
        ],
    ),
    QualityTestCase(
        name="find_function_usage_embedder_encode",
        file_pattern="scope.py",
        query="embedder.encode usage embedding",
        questions=[
            "How is embedder.encode used?",
            "What is passed to embedder.encode?",
        ],
        expected_answers=[
            "embedder.encode is called to get embeddings for query and chunks",
            "A list of texts (query + chunk texts) is passed",
        ],
        required_content=[
            "embedder.encode",
            "texts = [query]",
            "embeddings",
        ],
        nice_to_have=[
            "query_embedding",
            "chunk_embeddings",
        ],
    ),
    QualityTestCase(
        name="find_function_usage_np_dot",
        file_pattern="scope.py",
        query="numpy dot product cosine similarity calculation",
        questions=[
            "How is cosine similarity computed?",
            "What numpy functions are used?",
        ],
        expected_answers=[
            "Cosine similarity uses np.dot divided by the product of norms",
            "np.dot and np.linalg.norm are used",
        ],
        required_content=[
            "np.dot",
            "np.linalg.norm",
        ],
        nice_to_have=[
            "similarity",
            "query_embedding",
            "chunk_emb",
        ],
    ),
]


class QualityRunner:
    """Benchmark whether SCOPE preserves information needed for LLM tasks."""

    def __init__(
        self,
        config: BenchmarkConfig,
        embedder: Embedder | None = None,
        llm_evaluator: LLMEvaluator | None = None,
    ):
        """Initialize quality runner.

        Args:
            config: Benchmark configuration
            embedder: Embedder to use. Creates one from config if None.
            llm_evaluator: LLM evaluator for live evaluation. None for string-matching only.
        """
        self.config = config
        self.embedder = embedder or create_embedder(config.use_real_models)
        self.llm_evaluator = llm_evaluator

    def run(self, files: list[CorpusFile]) -> list[BenchmarkResult]:
        """Run quality benchmarks comparing SCOPE vs quick_compress.

        For each test case:
        1. Compress with SCOPE using the query
        2. Compress with quick_compress (no query awareness)
        3. Check which preserves more required content
        4. If LLM evaluator provided, also test with live LLM

        Args:
            files: List of corpus files to benchmark

        Returns:
            List of benchmark results
        """
        results: list[BenchmarkResult] = []

        for test_case in QUALITY_TEST_CASES:
            # Find matching file
            matching_files = [f for f in files if test_case.file_pattern in str(f.path)]

            for file in matching_files:
                for budget in self.config.budget_tokens:
                    # Run SCOPE compression
                    scope_result = self._evaluate_compression(
                        file=file,
                        test_case=test_case,
                        budget=budget,
                        method="scope",
                    )
                    results.append(scope_result)

                    # Run quick compression
                    quick_result = self._evaluate_compression(
                        file=file,
                        test_case=test_case,
                        budget=budget,
                        method="quick",
                    )
                    results.append(quick_result)

        return results

    def _evaluate_compression(
        self,
        file: CorpusFile,
        test_case: QualityTestCase,
        budget: int,
        method: str,
    ) -> BenchmarkResult:
        """Evaluate compression quality for a test case.

        Args:
            file: File to compress
            test_case: Test case with required content
            budget: Token budget
            method: "scope" or "quick"

        Returns:
            BenchmarkResult with quality metrics
        """
        original_tokens = file.estimated_tokens

        # Compress the content
        start = time.perf_counter()
        if method == "scope":
            compressed, symbol_index = compress_with_scope_indexed(
                text=file.content,
                query=test_case.query,
                budget_tokens=budget,
                embedder=self.embedder,
                summarizer=None,
                file_type=file.file_type,
                file_path=str(file.path),  # Enable AST-based chunking
            )
        else:
            compressed, symbol_index = quick_compress_indexed(
                text=file.content,
                budget_chars=budget * 4,
                file_type=file.file_type,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        compressed_tokens = len(compressed) // 4

        # Check content preservation (string matching - fast)
        required_preserved = 0
        required_missing = []
        for content in test_case.required_content:
            if content.lower() in compressed.lower():
                required_preserved += 1
            else:
                required_missing.append(content)

        nice_preserved = 0
        for content in test_case.nice_to_have:
            if content.lower() in compressed.lower():
                nice_preserved += 1

        total_required = len(test_case.required_content)
        total_nice = len(test_case.nice_to_have)

        # Calculate string-matching quality scores
        required_retention = required_preserved / total_required if total_required else 1.0
        nice_retention = nice_preserved / total_nice if total_nice else 1.0
        string_match_quality = (required_retention * 0.7) + (nice_retention * 0.3)

        # LLM evaluation (if evaluator provided)
        llm_accuracy = None
        llm_evaluations = None
        if self.llm_evaluator:
            try:
                llm_result = self.llm_evaluator.evaluate(
                    content=compressed,
                    questions=test_case.questions,
                    expected_answers=test_case.expected_answers,
                    context_description=f"Python {file.file_type} file",
                )
                llm_accuracy = llm_result.accuracy
                llm_evaluations = [
                    {
                        "question": e.question,
                        "expected": e.expected_answer,
                        "actual": e.llm_answer,
                        "correct": e.is_correct,
                        "confidence": e.confidence,
                    }
                    for e in llm_result.evaluations
                ]
            except Exception as e:
                # Log error but continue with string matching
                llm_evaluations = [{"error": str(e)}]

        # Overall quality: use LLM accuracy if available, otherwise string matching
        overall_quality = llm_accuracy if llm_accuracy is not None else string_match_quality

        # LLM answerability: use LLM result if available
        if llm_accuracy is not None:
            answerable = llm_accuracy >= 0.5  # At least half questions answered correctly
        else:
            answerable = required_preserved == total_required

        quality_data = {
            "required_preserved": required_preserved,
            "required_total": total_required,
            "required_retention": required_retention,
            "required_missing": required_missing,
            "nice_preserved": nice_preserved,
            "nice_total": total_nice,
            "nice_retention": nice_retention,
            "string_match_quality": string_match_quality,
            "llm_accuracy": llm_accuracy,
            "llm_evaluations": llm_evaluations,
            "overall_quality": overall_quality,
            "questions_answerable": answerable,
            "questions": test_case.questions,
            "evaluation_mode": "llm" if llm_accuracy is not None else "string_match",
        }

        return BenchmarkResult(
            scenario_name=f"quality_{test_case.name}_{budget}",
            file_path=str(file.path),
            file_type=file.file_type,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=(compressed_tokens / original_tokens if original_tokens else 1.0),
            budget_tokens=budget,
            budget_adherence=compressed_tokens / budget if budget else 0.0,
            elapsed_ms=elapsed_ms,
            query=test_case.query,
            method=method,
            symbol_count=symbol_index.count("- L") if symbol_index else 0,
            extra_data=quality_data,
        )

    def get_quality_summary(self, results: list[BenchmarkResult]) -> dict:
        """Calculate quality comparison between SCOPE and quick.

        Args:
            results: List of quality benchmark results

        Returns:
            Dictionary with comparison metrics
        """
        scope_results = [r for r in results if r.method == "scope"]
        quick_results = [r for r in results if r.method == "quick"]

        if not scope_results or not quick_results:
            return {"comparison_available": False}

        # Compare by test case and budget
        scope_wins = 0
        quick_wins = 0
        ties = 0

        scope_answerable = 0
        quick_answerable = 0

        scope_quality_scores = []
        quick_quality_scores = []

        scope_llm_scores = []
        quick_llm_scores = []

        evaluation_mode = "string_match"

        for scope_r in scope_results:
            # Find matching quick result
            for quick_r in quick_results:
                if (
                    scope_r.file_path == quick_r.file_path
                    and scope_r.budget_tokens == quick_r.budget_tokens
                    and scope_r.scenario_name == quick_r.scenario_name.replace("quick", "scope")
                ):
                    scope_quality = scope_r.extra_data["overall_quality"]
                    quick_quality = quick_r.extra_data["overall_quality"]

                    scope_quality_scores.append(scope_quality)
                    quick_quality_scores.append(quick_quality)

                    # Track LLM scores separately if available
                    if scope_r.extra_data.get("llm_accuracy") is not None:
                        evaluation_mode = "llm"
                        scope_llm_scores.append(scope_r.extra_data["llm_accuracy"])
                        quick_llm_scores.append(quick_r.extra_data.get("llm_accuracy", 0))

                    if scope_quality > quick_quality:
                        scope_wins += 1
                    elif quick_quality > scope_quality:
                        quick_wins += 1
                    else:
                        ties += 1

                    if scope_r.extra_data["questions_answerable"]:
                        scope_answerable += 1
                    if quick_r.extra_data["questions_answerable"]:
                        quick_answerable += 1
                    break

        total = scope_wins + quick_wins + ties

        summary = {
            "comparison_available": True,
            "evaluation_mode": evaluation_mode,
            "scope_quality_wins": scope_wins,
            "quick_quality_wins": quick_wins,
            "ties": ties,
            "scope_win_rate": scope_wins / total if total else 0,
            "scope_avg_quality": (
                sum(scope_quality_scores) / len(scope_quality_scores) if scope_quality_scores else 0
            ),
            "quick_avg_quality": (
                sum(quick_quality_scores) / len(quick_quality_scores) if quick_quality_scores else 0
            ),
            "scope_questions_answerable": scope_answerable,
            "quick_questions_answerable": quick_answerable,
            "total_comparisons": total,
        }

        # Add LLM-specific metrics if available
        if scope_llm_scores:
            summary["scope_avg_llm_accuracy"] = sum(scope_llm_scores) / len(scope_llm_scores)
            summary["quick_avg_llm_accuracy"] = (
                sum(quick_llm_scores) / len(quick_llm_scores) if quick_llm_scores else 0
            )

        return summary
