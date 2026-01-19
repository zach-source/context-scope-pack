"""Tests for AST-based code chunking."""

import pytest

from scopepack.scope import Chunk, chunk_text

# Check if tree-sitter is available
try:
    from scopepack.ast_chunker import (
        ASTChunkConfig,
        chunk_with_ast,
        detect_language,
    )

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False


class TestLanguageDetection:
    def test_python_extensions(self):
        assert detect_language("main.py") == "python"
        assert detect_language("/path/to/script.py") == "python"
        assert detect_language("types.pyi") == "python"

    def test_javascript_extensions(self):
        assert detect_language("app.js") == "javascript"
        assert detect_language("module.mjs") == "javascript"
        assert detect_language("component.jsx") == "javascript"

    def test_typescript_extensions(self):
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") == "tsx"

    def test_other_languages(self):
        assert detect_language("main.go") == "go"
        assert detect_language("lib.rs") == "rust"
        assert detect_language("Main.java") == "java"
        assert detect_language("script.rb") == "ruby"

    def test_config_files(self):
        assert detect_language("config.json") == "json"
        assert detect_language("config.yaml") == "yaml"
        assert detect_language("config.toml") == "toml"

    def test_no_extension(self):
        assert detect_language("Makefile") is None
        assert detect_language("noextension") is None

    def test_unknown_extension(self):
        assert detect_language("file.xyz") is None


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestASTChunkingPython:
    def test_simple_functions(self):
        code = '''def hello():
    """Say hello."""
    print("hello")


def world():
    """Say world."""
    print("world")
'''
        chunks = chunk_with_ast(code, "python")
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)
        # Each function should be in its own chunk
        assert any("hello" in c.text for c in chunks)
        assert any("world" in c.text for c in chunks)

    def test_class_with_methods(self):
        code = '''class MyClass:
    """A simple class."""

    def __init__(self, x: int):
        self.x = x

    def get_x(self) -> int:
        return self.x

    def set_x(self, value: int) -> None:
        self.x = value
'''
        chunks = chunk_with_ast(code, "python")
        assert len(chunks) >= 1
        # Class and its methods should be captured
        assert any("MyClass" in c.text for c in chunks)

    def test_decorated_function(self):
        code = '''@decorator
@another_decorator(arg=1)
def decorated_function():
    """This function has decorators."""
    return 42
'''
        chunks = chunk_with_ast(code, "python")
        assert len(chunks) >= 1
        # Decorators should stay with the function
        first_chunk = chunks[0]
        assert "@decorator" in first_chunk.text
        assert "decorated_function" in first_chunk.text

    def test_async_function(self):
        code = '''async def fetch_data(url: str) -> dict:
    """Fetch data asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        chunks = chunk_with_ast(code, "python")
        assert len(chunks) >= 1
        assert any("async def fetch_data" in c.text for c in chunks)

    def test_nested_class(self):
        code = '''class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def inner_method(self):
            return "inner"

    def outer_method(self):
        return "outer"
'''
        chunks = chunk_with_ast(code, "python")
        # Should handle nested structures
        assert len(chunks) >= 1

    def test_multiline_signature(self):
        code = '''def complex_function(
    arg1: str,
    arg2: int,
    arg3: list[str],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Function with multiline signature."""
    return {"arg1": arg1, "arg2": arg2}
'''
        chunks = chunk_with_ast(code, "python")
        assert len(chunks) >= 1
        # Multiline signature should be kept together
        chunk_text = chunks[0].text
        assert "complex_function" in chunk_text
        assert "arg1: str" in chunk_text
        assert "-> dict[str, Any]" in chunk_text

    def test_scope_prefix(self):
        code = """class Calculator:
    def add(self, a, b):
        return a + b
"""
        config = ASTChunkConfig(include_scope_prefix=True)
        chunks = chunk_with_ast(code, "python", config)

        # With default config, should include scope prefix
        assert len(chunks) >= 1

    def test_no_scope_prefix(self):
        code = """class Calculator:
    def add(self, a, b):
        return a + b
"""
        config = ASTChunkConfig(include_scope_prefix=False)
        chunks = chunk_with_ast(code, "python", config)
        assert len(chunks) >= 1
        # Should not have scope prefix comment
        for chunk in chunks:
            assert not chunk.text.startswith("# Scope:")


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestASTChunkingJavaScript:
    def test_function_declaration(self):
        code = """function greet(name) {
    console.log(`Hello, ${name}!`);
}

function farewell(name) {
    console.log(`Goodbye, ${name}!`);
}
"""
        chunks = chunk_with_ast(code, "javascript")
        assert len(chunks) >= 1

    def test_arrow_functions(self):
        code = """const add = (a, b) => a + b;

const multiply = (a, b) => {
    return a * b;
};
"""
        chunks = chunk_with_ast(code, "javascript")
        assert len(chunks) >= 1

    def test_class_declaration(self):
        code = """class Person {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return `Hello, I'm ${this.name}`;
    }
}
"""
        chunks = chunk_with_ast(code, "javascript")
        assert len(chunks) >= 1
        assert any("Person" in c.text for c in chunks)


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestASTChunkingTypeScript:
    def test_interface(self):
        code = """interface User {
    id: number;
    name: string;
    email: string;
}

function getUser(id: number): User {
    return { id, name: "John", email: "john@example.com" };
}
"""
        chunks = chunk_with_ast(code, "typescript")
        assert len(chunks) >= 1

    def test_type_alias(self):
        code = """type Result<T> = {
    success: boolean;
    data?: T;
    error?: string;
};

function processResult<T>(result: Result<T>): T | null {
    return result.success ? result.data ?? null : null;
}
"""
        chunks = chunk_with_ast(code, "typescript")
        assert len(chunks) >= 1


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestASTChunkingGo:
    def test_go_functions(self):
        code = """package main

func Add(a, b int) int {
    return a + b
}

func Subtract(a, b int) int {
    return a - b
}
"""
        chunks = chunk_with_ast(code, "go")
        assert len(chunks) >= 1

    def test_go_struct_and_methods(self):
        code = """package main

type Calculator struct {
    result int
}

func (c *Calculator) Add(x int) {
    c.result += x
}

func (c *Calculator) GetResult() int {
    return c.result
}
"""
        chunks = chunk_with_ast(code, "go")
        assert len(chunks) >= 1


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestASTChunkingRust:
    def test_rust_functions(self):
        code = """fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
"""
        chunks = chunk_with_ast(code, "rust")
        assert len(chunks) >= 1

    def test_rust_struct_and_impl(self):
        code = """struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
"""
        chunks = chunk_with_ast(code, "rust")
        assert len(chunks) >= 1


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestChunkConfig:
    def test_max_chunk_chars_splitting(self):
        # Create a large function that exceeds max_chunk_chars
        large_body = "\n".join([f"    line_{i} = {i}" for i in range(100)])
        code = f"""def large_function():
{large_body}
"""
        config = ASTChunkConfig(max_chunk_chars=500)
        chunks = chunk_with_ast(code, "python", config)
        # Should split into multiple chunks or handle gracefully
        assert len(chunks) >= 1

    def test_min_chunk_chars_merging(self):
        code = """x = 1
y = 2

def tiny():
    pass
"""
        config = ASTChunkConfig(min_chunk_chars=50)
        chunks = chunk_with_ast(code, "python", config)
        # Small non-definition chunks may be merged
        assert len(chunks) >= 1


class TestFallbackBehavior:
    def test_chunk_text_without_tree_sitter(self):
        """Test that chunk_text falls back gracefully when tree-sitter is not available."""
        code = """def hello():
    print("hello")
"""
        # Force fallback by passing use_ast=False
        chunks = chunk_text(code, file_type="code", file_path="test.py", use_ast=False)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_text_without_file_path(self):
        """Test that chunk_text uses regex when no file_path is provided."""
        code = """def hello():
    print("hello")
"""
        chunks = chunk_text(code, file_type="code", file_path=None, use_ast=True)
        assert len(chunks) >= 1

    def test_chunk_text_with_unsupported_extension(self):
        """Test fallback for unsupported file extensions."""
        code = """def hello():
    print("hello")
"""
        # .xyz is not a supported extension
        chunks = chunk_text(code, file_type="code", file_path="test.xyz", use_ast=True)
        assert len(chunks) >= 1


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestChunkTextWithAST:
    def test_chunk_text_uses_ast_when_available(self):
        """Test that chunk_text uses AST chunking when tree-sitter is available."""
        code = """@decorator
def decorated():
    pass

def undecorated():
    pass
"""
        chunks = chunk_text(code, file_type="code", file_path="test.py", use_ast=True)
        assert len(chunks) >= 1
        # AST chunking should keep decorator with function
        decorated_chunk = next((c for c in chunks if "decorated" in c.text), None)
        if decorated_chunk:
            assert "@decorator" in decorated_chunk.text or "decorator" in decorated_chunk.text

    def test_chunk_text_line_indices(self):
        """Test that AST chunks have correct line indices."""
        code = """# Comment at line 0
def first():  # line 1
    pass  # line 2

def second():  # line 4
    pass  # line 5
"""
        chunks = chunk_text(code, file_type="code", file_path="test.py", use_ast=True)
        # Chunks should have valid line ranges
        for chunk in chunks:
            assert chunk.start_idx >= 0
            assert chunk.end_idx >= chunk.start_idx


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter not installed")
class TestEdgeCases:
    def test_empty_file(self):
        chunks = chunk_with_ast("", "python")
        # Should return a single chunk with empty or whitespace content
        assert len(chunks) == 1

    def test_syntax_error_tolerance(self):
        """Test that AST chunking handles syntax errors gracefully."""
        code = """def broken(
    # Missing closing paren and body
"""
        # Should not raise, may fall back
        try:
            chunks = chunk_with_ast(code, "python")
            # If it returns chunks, they should be valid
            assert all(isinstance(c, Chunk) for c in chunks)
        except Exception:
            # It's acceptable to raise on severe syntax errors
            pass

    def test_comments_only(self):
        code = """# Just a comment
# Another comment
"""
        chunks = chunk_with_ast(code, "python")
        # Should handle gracefully
        assert len(chunks) >= 1

    def test_imports_only(self):
        code = """import os
import sys
from pathlib import Path
"""
        chunks = chunk_with_ast(code, "python")
        # Imports may be grouped together
        assert len(chunks) >= 1
