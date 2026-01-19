"""Test corpus discovery and management for benchmarks."""

from dataclasses import dataclass
from pathlib import Path

from scopepack.scope import detect_file_type


@dataclass
class CorpusFile:
    """A file in the test corpus."""

    path: Path
    file_type: str
    size_bytes: int
    content: str

    @property
    def size_category(self) -> str:
        """Categorize file by size for performance benchmarks."""
        if self.size_bytes < 1024:
            return "tiny"  # <1KB
        elif self.size_bytes < 5 * 1024:
            return "small"  # 1-5KB
        elif self.size_bytes < 20 * 1024:
            return "medium"  # 5-20KB
        else:
            return "large"  # >20KB

    @property
    def estimated_tokens(self) -> int:
        """Estimate token count (4 chars per token)."""
        return len(self.content) // 4


class TestCorpus:
    """Discovers and manages test files for benchmarking."""

    # Default patterns for corpus discovery
    DEFAULT_PATTERNS = [
        "src/scopepack/*.py",
        ".claude/hooks/*.py",
        "*.md",
        "pyproject.toml",
        "devenv.nix",
    ]

    def __init__(self, root_dir: Path):
        """Initialize corpus with root directory.

        Args:
            root_dir: Project root directory to discover files from
        """
        self.root_dir = Path(root_dir)
        self._files: list[CorpusFile] = []
        self._loaded = False

    def discover(self, patterns: list[str] | None = None) -> list[CorpusFile]:
        """Discover test files matching patterns.

        Args:
            patterns: Glob patterns to match. Uses DEFAULT_PATTERNS if None.

        Returns:
            List of discovered CorpusFile objects
        """
        if patterns is None:
            patterns = self.DEFAULT_PATTERNS

        self._files = []
        seen_paths: set[Path] = set()

        for pattern in patterns:
            for path in self.root_dir.glob(pattern):
                if path.is_file() and path not in seen_paths:
                    seen_paths.add(path)
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        self._files.append(
                            CorpusFile(
                                path=path,
                                file_type=detect_file_type(str(path)),
                                size_bytes=len(content.encode("utf-8")),
                                content=content,
                            )
                        )
                    except Exception:
                        # Skip files that can't be read
                        continue

        self._loaded = True
        return self._files

    @property
    def files(self) -> list[CorpusFile]:
        """Get discovered files, discovering if not already done."""
        if not self._loaded:
            self.discover()
        return self._files

    def by_type(self, file_type: str) -> list[CorpusFile]:
        """Filter corpus files by type."""
        return [f for f in self.files if f.file_type == file_type]

    def by_size_category(self, category: str) -> list[CorpusFile]:
        """Filter corpus files by size category."""
        return [f for f in self.files if f.size_category == category]

    def largest_files(self, n: int = 5) -> list[CorpusFile]:
        """Get the N largest files in the corpus."""
        return sorted(self.files, key=lambda f: f.size_bytes, reverse=True)[:n]

    def get_summary(self) -> dict:
        """Get summary statistics about the corpus."""
        files = self.files
        if not files:
            return {"total_files": 0}

        by_type: dict[str, int] = {}
        by_size: dict[str, int] = {}
        total_bytes = 0
        total_tokens = 0

        for f in files:
            by_type[f.file_type] = by_type.get(f.file_type, 0) + 1
            by_size[f.size_category] = by_size.get(f.size_category, 0) + 1
            total_bytes += f.size_bytes
            total_tokens += f.estimated_tokens

        return {
            "total_files": len(files),
            "total_bytes": total_bytes,
            "total_estimated_tokens": total_tokens,
            "by_type": by_type,
            "by_size_category": by_size,
            "largest_file": str(self.largest_files(1)[0].path) if files else None,
        }


def create_default_corpus(project_root: Path | None = None) -> TestCorpus:
    """Create a corpus with the default project files.

    Args:
        project_root: Project root path. Auto-detects if None.

    Returns:
        TestCorpus configured for this project
    """
    if project_root is None:
        # Auto-detect project root by looking for pyproject.toml
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                project_root = current
                break
            current = current.parent
        else:
            project_root = Path.cwd()

    return TestCorpus(project_root)
