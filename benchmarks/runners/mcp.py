"""MCP output compression benchmarks - measure SCOPE effectiveness on live MCP results.

Tests whether SCOPE compression preserves information from real MCP tool outputs
(like Context7 library documentation) while reducing token usage.

Uses live LLM evaluation to measure:
1. Original docs → can LLM answer questions?
2. SCOPE compressed → can LLM still answer questions?
3. Quick compressed → can LLM answer questions?
"""

import time
from dataclasses import dataclass

from benchmarks.config import BenchmarkConfig, BenchmarkResult
from benchmarks.llm_evaluator import LLMEvaluator
from benchmarks.mock_embedder import create_embedder
from scopepack.mcp_utils import compress_mcp_result
from scopepack.scope import Embedder, quick_compress_indexed


@dataclass
class MCPTestCase:
    """A test case for measuring compression quality on MCP output."""

    name: str
    library_id: str  # Context7 library ID (e.g., "/vercel/next.js")
    topic: str  # Topic to fetch docs for
    questions: list[str]  # Questions the LLM should answer from the docs
    expected_answers: list[str]  # Expected answers for judging
    tokens_to_fetch: int = 5000  # How many tokens of docs to fetch


# Test cases covering different library documentation types
MCP_TEST_CASES = [
    MCPTestCase(
        name="react_hooks",
        library_id="/facebook/react",
        topic="useState useEffect hooks",
        questions=[
            "What does useState return?",
            "When does useEffect run?",
            "How do you skip an effect on re-renders?",
        ],
        expected_answers=[
            "useState returns a pair: the current state value and a function to update it",
            "useEffect runs after the render is committed to the screen, after every completed render by default",
            "Pass an empty dependency array [] to skip re-running the effect, or specific dependencies to only run when they change",
        ],
        tokens_to_fetch=5000,
    ),
    MCPTestCase(
        name="fastapi_routing",
        library_id="/fastapi/fastapi",
        topic="path operations routing decorators",
        questions=[
            "How do you define a GET endpoint in FastAPI?",
            "How do you get path parameters?",
            "What decorators are available for HTTP methods?",
        ],
        expected_answers=[
            "Use @app.get('/path') decorator on a function",
            "Define path parameters in the route like /items/{item_id} and as function parameters",
            "@app.get, @app.post, @app.put, @app.delete, @app.patch, @app.options, @app.head",
        ],
        tokens_to_fetch=5000,
    ),
    MCPTestCase(
        name="nextjs_routing",
        library_id="/vercel/next.js",
        topic="app router pages routing",
        questions=[
            "How does file-based routing work in Next.js App Router?",
            "What is a layout.js file for?",
            "How do you create dynamic routes?",
        ],
        expected_answers=[
            "Files in the app directory become routes - page.js defines the UI for a route segment",
            "layout.js defines shared UI that wraps child routes and preserves state across navigations",
            "Use square brackets for dynamic segments like [id] or [slug] in folder names",
        ],
        tokens_to_fetch=5000,
    ),
    MCPTestCase(
        name="pandas_dataframes",
        library_id="/pandas-dev/pandas",
        topic="DataFrame operations selection filtering",
        questions=[
            "How do you select a column from a DataFrame?",
            "How do you filter rows based on a condition?",
            "What is the difference between loc and iloc?",
        ],
        expected_answers=[
            "Use df['column_name'] or df.column_name to select a single column",
            "Use boolean indexing like df[df['column'] > value] to filter rows",
            "loc uses labels/names for selection, iloc uses integer positions/indices",
        ],
        tokens_to_fetch=5000,
    ),
    MCPTestCase(
        name="anthropic_sdk",
        library_id="/anthropics/anthropic-sdk-python",
        topic="messages API create completion",
        questions=[
            "How do you create a message with the Anthropic SDK?",
            "What parameters are required for client.messages.create?",
            "How do you handle streaming responses?",
        ],
        expected_answers=[
            "Use client.messages.create() with model, max_tokens, and messages parameters",
            "model (string), max_tokens (int), and messages (list of message dicts with role and content)",
            "Use client.messages.stream() or pass stream=True to get streaming responses",
        ],
        tokens_to_fetch=5000,
    ),
]


class MCPRunner:
    """Benchmark SCOPE compression on live MCP tool outputs."""

    def __init__(
        self,
        config: BenchmarkConfig,
        embedder: Embedder | None = None,
        llm_evaluator: LLMEvaluator | None = None,
    ):
        """Initialize MCP benchmark runner.

        Args:
            config: Benchmark configuration
            embedder: Embedder to use for SCOPE compression
            llm_evaluator: LLM evaluator for quality assessment (required)
        """
        self.config = config
        self.embedder = embedder or create_embedder(config.use_real_models)
        self.llm_evaluator = llm_evaluator
        self._context7_available: bool | None = None

    def _check_context7(self) -> bool:
        """Check if Context7 MCP is available."""
        if self._context7_available is not None:
            return self._context7_available

        try:
            import httpx

            # Try to resolve a known library
            response = httpx.post(
                "http://localhost:3000/resolve-library-id",  # Default Context7 port
                json={"libraryName": "react"},
                timeout=5.0,
            )
            self._context7_available = response.status_code == 200
        except Exception:
            # Try via MCP tool call if available
            self._context7_available = False

        return self._context7_available

    def _fetch_context7_docs(
        self,
        library_id: str,
        topic: str,
        tokens: int,
    ) -> str | None:
        """Fetch documentation from Context7.

        Args:
            library_id: Context7 library ID
            topic: Topic to focus on
            tokens: Max tokens to fetch

        Returns:
            Documentation content or None if failed
        """
        try:
            import httpx

            # Call Context7 get-library-docs
            response = httpx.post(
                "http://localhost:3000/get-library-docs",
                json={
                    "context7CompatibleLibraryID": library_id,
                    "topic": topic,
                    "tokens": tokens,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("content", "")
        except Exception as e:
            print(f"  Context7 fetch failed: {e}")

        return None

    def run(self, _files=None) -> list[BenchmarkResult]:
        """Run MCP compression benchmarks.

        For each test case:
        1. Fetch docs from Context7
        2. Have LLM answer questions with original docs
        3. Compress with SCOPE and test again
        4. Compress with quick_compress and test again
        5. Compare results

        Args:
            _files: Ignored (MCP runner doesn't use corpus files)

        Returns:
            List of benchmark results
        """
        results: list[BenchmarkResult] = []

        if not self.llm_evaluator:
            print("  WARNING: MCP benchmarks require --llm-eval flag")
            return results

        for test_case in MCP_TEST_CASES:
            print(f"\n  Testing: {test_case.name}")
            print(f"    Library: {test_case.library_id}")
            print(f"    Topic: {test_case.topic}")

            # Fetch docs from Context7
            docs = self._fetch_context7_docs(
                library_id=test_case.library_id,
                topic=test_case.topic,
                tokens=test_case.tokens_to_fetch,
            )

            if not docs:
                print(f"    SKIP: Could not fetch docs for {test_case.library_id}")
                continue

            original_tokens = len(docs) // 4
            print(f"    Fetched: {original_tokens} tokens")

            # Test with original docs
            original_result = self._evaluate_with_llm(
                content=docs,
                test_case=test_case,
                method="original",
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
            )
            results.append(original_result)
            print(f"    Original accuracy: {original_result.extra_data.get('llm_accuracy', 0):.0%}")

            # Test with SCOPE compression at different budgets
            for budget in self.config.budget_tokens:
                # SCOPE compression
                start = time.perf_counter()
                compressed = compress_mcp_result(
                    content=docs,
                    query=test_case.topic,
                    source="context7",
                    budget_tokens=budget,
                    threshold_tokens=100,  # Always compress for benchmark
                    use_daemon=False,  # Use direct library
                    include_metadata=False,
                )
                scope_elapsed = (time.perf_counter() - start) * 1000
                scope_tokens = len(compressed) // 4

                scope_result = self._evaluate_with_llm(
                    content=compressed,
                    test_case=test_case,
                    method="scope",
                    original_tokens=original_tokens,
                    compressed_tokens=scope_tokens,
                    budget=budget,
                    elapsed_ms=scope_elapsed,
                )
                results.append(scope_result)

                # Quick compression
                start = time.perf_counter()
                quick_compressed, _ = quick_compress_indexed(
                    docs,
                    budget_chars=budget * 4,
                    file_type="prose",
                )
                quick_elapsed = (time.perf_counter() - start) * 1000
                quick_tokens = len(quick_compressed) // 4

                quick_result = self._evaluate_with_llm(
                    content=quick_compressed,
                    test_case=test_case,
                    method="quick",
                    original_tokens=original_tokens,
                    compressed_tokens=quick_tokens,
                    budget=budget,
                    elapsed_ms=quick_elapsed,
                )
                results.append(quick_result)

                print(
                    f"    Budget {budget}: SCOPE={scope_result.extra_data.get('llm_accuracy', 0):.0%} "
                    f"({scope_tokens}t), Quick={quick_result.extra_data.get('llm_accuracy', 0):.0%} "
                    f"({quick_tokens}t)"
                )

        return results

    def _evaluate_with_llm(
        self,
        content: str,
        test_case: MCPTestCase,
        method: str,
        original_tokens: int,
        compressed_tokens: int,
        budget: int | None = None,
        elapsed_ms: float = 0,
    ) -> BenchmarkResult:
        """Evaluate compression with LLM.

        Args:
            content: Content to evaluate
            test_case: Test case with questions
            method: "original", "scope", or "quick"
            original_tokens: Original token count
            compressed_tokens: Compressed token count
            budget: Token budget (None for original)
            elapsed_ms: Compression time

        Returns:
            BenchmarkResult with LLM evaluation
        """
        llm_accuracy = 0.0
        llm_evaluations = []

        if self.llm_evaluator:
            try:
                llm_result = self.llm_evaluator.evaluate(
                    content=content,
                    questions=test_case.questions,
                    expected_answers=test_case.expected_answers,
                    context_description=f"library documentation for {test_case.library_id}",
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
                llm_evaluations = [{"error": str(e)}]

        return BenchmarkResult(
            scenario_name=f"mcp_{test_case.name}_{method}_{budget or 'full'}",
            file_path=test_case.library_id,
            file_type="mcp_docs",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens else 1.0,
            budget_tokens=budget or original_tokens,
            budget_adherence=compressed_tokens / budget if budget else 1.0,
            elapsed_ms=elapsed_ms,
            query=test_case.topic,
            method=method,
            symbol_count=0,
            extra_data={
                "library_id": test_case.library_id,
                "topic": test_case.topic,
                "llm_accuracy": llm_accuracy,
                "llm_evaluations": llm_evaluations,
                "questions": test_case.questions,
                "questions_total": len(test_case.questions),
                "questions_correct": int(llm_accuracy * len(test_case.questions)),
            },
        )

    def get_mcp_summary(self, results: list[BenchmarkResult]) -> dict:
        """Calculate MCP benchmark summary.

        Args:
            results: List of MCP benchmark results

        Returns:
            Dictionary with summary metrics
        """
        original_results = [r for r in results if r.method == "original"]
        scope_results = [r for r in results if r.method == "scope"]
        quick_results = [r for r in results if r.method == "quick"]

        def avg_accuracy(results_list):
            accuracies = [r.extra_data.get("llm_accuracy", 0) for r in results_list]
            return sum(accuracies) / len(accuracies) if accuracies else 0

        def avg_tokens(results_list):
            tokens = [r.compressed_tokens for r in results_list]
            return sum(tokens) / len(tokens) if tokens else 0

        # Calculate quality retention (how much of original accuracy is preserved)
        original_acc = avg_accuracy(original_results)
        scope_acc = avg_accuracy(scope_results)
        quick_acc = avg_accuracy(quick_results)

        scope_retention = scope_acc / original_acc if original_acc > 0 else 0
        quick_retention = quick_acc / original_acc if original_acc > 0 else 0

        # Token reduction
        original_tokens = avg_tokens(original_results)
        scope_tokens = avg_tokens(scope_results)
        quick_tokens = avg_tokens(quick_results)

        scope_reduction = 1 - (scope_tokens / original_tokens) if original_tokens > 0 else 0
        quick_reduction = 1 - (quick_tokens / original_tokens) if original_tokens > 0 else 0

        # Win rate (SCOPE vs Quick at same budget)
        scope_wins = 0
        quick_wins = 0
        ties = 0

        for scope_r in scope_results:
            for quick_r in quick_results:
                if (
                    scope_r.extra_data.get("library_id") == quick_r.extra_data.get("library_id")
                    and scope_r.budget_tokens == quick_r.budget_tokens
                ):
                    scope_q = scope_r.extra_data.get("llm_accuracy", 0)
                    quick_q = quick_r.extra_data.get("llm_accuracy", 0)
                    if scope_q > quick_q:
                        scope_wins += 1
                    elif quick_q > scope_q:
                        quick_wins += 1
                    else:
                        ties += 1
                    break

        total = scope_wins + quick_wins + ties

        return {
            "test_cases": len(original_results),
            "original_avg_accuracy": original_acc,
            "scope_avg_accuracy": scope_acc,
            "quick_avg_accuracy": quick_acc,
            "scope_quality_retention": scope_retention,
            "quick_quality_retention": quick_retention,
            "original_avg_tokens": original_tokens,
            "scope_avg_tokens": scope_tokens,
            "quick_avg_tokens": quick_tokens,
            "scope_token_reduction": scope_reduction,
            "quick_token_reduction": quick_reduction,
            "scope_wins": scope_wins,
            "quick_wins": quick_wins,
            "ties": ties,
            "scope_win_rate": scope_wins / total if total > 0 else 0,
            "total_comparisons": total,
        }
