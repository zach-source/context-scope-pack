"""ScopePack daemon - HTTP server for compression operations.

Loads models once and serves compression requests via HTTP.
This avoids the per-hook model loading overhead.

Supports multiple embedding providers via SCOPE_EMBEDDER env var:
- Local: bge-small-en-v1.5 (default, fast)
- Bedrock Titan: titan-embed-text-v2:256/512/1024
- Bedrock Cohere: cohere-embed-english-v3, cohere-embed-multilingual-v3
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .db import CacheDB, CacheEntry, FileSummary, content_hash, init_db
from .embedders import create_embedder
from .scope import (
    compress_with_scope_indexed,
    detect_file_type,
    quick_compress,
    quick_compress_indexed,
)

# Configuration
HOST = os.environ.get("SCOPE_DAEMON_HOST", "127.0.0.1")
PORT = int(os.environ.get("SCOPE_DAEMON_PORT", "18765"))
EMBEDDER_TYPE = os.environ.get("SCOPE_EMBEDDER", "bge-small-en-v1.5")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE")
MODEL_VERSION = f"v2-{EMBEDDER_TYPE}"  # Version includes embedder type

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scopepack.daemon")


class Models:
    """Lazy-loaded ML models."""

    _embedder = None
    _summarizer = None
    _tokenizer = None
    _loading = False
    _embedder_type = None

    @classmethod
    async def get_embedder(cls):
        """Get or load the embedding model.

        Uses SCOPE_EMBEDDER env var to select model type:
        - bge-small-en-v1.5 (default, local)
        - titan-embed-text-v2:256/512/1024 (Bedrock)
        - cohere-embed-english-v3 (Bedrock)
        - cohere-embed-multilingual-v3 (Bedrock)
        """
        if cls._embedder is None and not cls._loading:
            cls._loading = True
            try:
                logger.info(f"Loading embedding model: {EMBEDDER_TYPE}...")
                loop = asyncio.get_event_loop()

                # Create embedder based on type
                cls._embedder = await loop.run_in_executor(
                    None,
                    lambda: create_embedder(
                        EMBEDDER_TYPE,
                        region=AWS_REGION,
                        profile=AWS_PROFILE,
                    ),
                )
                cls._embedder_type = EMBEDDER_TYPE
                logger.info(
                    f"Embedding model loaded: {EMBEDDER_TYPE} ({cls._embedder.dimensions} dims)"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                cls._loading = False
                raise
        return cls._embedder

    @classmethod
    async def get_summarizer(cls):
        """Get or load the summarization model."""
        if cls._summarizer is None and not cls._loading:
            try:
                logger.info("Loading summarization model...")
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                loop = asyncio.get_event_loop()

                def load_summarizer():
                    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
                    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
                    return model, tokenizer

                cls._summarizer, cls._tokenizer = await loop.run_in_executor(None, load_summarizer)
                logger.info("Summarization model loaded")
            except Exception as e:
                logger.error(f"Failed to load summarization model: {e}")
                raise
        return cls._summarizer, cls._tokenizer


class SummarizerWrapper:
    """Wrapper to match the Summarizer protocol."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def summarize(self, text: str, max_length: int) -> str:
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Database
db = CacheDB()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown logic."""
    # Initialize database
    await init_db()
    logger.info(f"ScopePack daemon starting on {HOST}:{PORT}")

    # Optionally preload models (can be slow)
    if os.environ.get("SCOPE_PRELOAD_MODELS", "0") == "1":
        logger.info("Preloading models...")
        await Models.get_embedder()
        await Models.get_summarizer()
        logger.info("Models preloaded")

    yield

    logger.info("ScopePack daemon shutting down")


app = FastAPI(
    title="ScopePack Daemon",
    description="Token-efficient context compression for Claude Code",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class CompressRequest(BaseModel):
    """Request to compress content."""

    text: str = Field(..., description="Text to compress")
    query: str | None = Field(None, description="Query for relevance scoring")
    budget_tokens: int = Field(900, description="Target token budget")
    file_path: str | None = Field(None, description="File path for type detection")
    use_cache: bool = Field(True, description="Whether to use cache")


class CompressResponse(BaseModel):
    """Compressed content response."""

    compressed_text: str
    symbol_index: str | None = None  # Line-numbered symbol index for navigation
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    cache_hit: bool
    model_version: str


class SummarizeRequest(BaseModel):
    """Request to summarize content."""

    text: str = Field(..., description="Text to summarize")
    max_tokens: int = Field(150, description="Max summary tokens")


class SummarizeResponse(BaseModel):
    """Summary response."""

    summary: str
    original_tokens: int
    summary_tokens: int


class FileSummaryRequest(BaseModel):
    """Request for file summary."""

    file_path: str
    file_hash: str
    content: str | None = None  # If not provided, daemon reads file


class FileSummaryResponse(BaseModel):
    """File summary response."""

    summary: str
    summary_tokens: int
    cache_hit: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: dict[str, bool]
    cache_stats: dict[str, Any]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded={
            "embedder": Models._embedder is not None,
            "summarizer": Models._summarizer is not None,
        },
        cache_stats={
            "db_path": str(db.db_path),
        },
    )


@app.post("/compress", response_model=CompressResponse)
async def compress(request: CompressRequest):
    """Compress content using SCOPE algorithm."""
    text = request.text
    query = request.query or ""
    budget = request.budget_tokens

    # Estimate tokens (rough: 4 chars per token)
    original_tokens = len(text) // 4

    # Check cache first
    text_hash = content_hash(text)
    query_hash = content_hash(query) if query else None

    if request.use_cache:
        cached = await db.get_compressed(text_hash, query_hash, budget, MODEL_VERSION)
        if cached:
            await db.record_stats(
                hit=True,
                tokens_saved=original_tokens - (cached.compressed_tokens or 0),
            )
            return CompressResponse(
                compressed_text=cached.compressed_text,
                original_tokens=cached.original_tokens or original_tokens,
                compressed_tokens=cached.compressed_tokens or len(cached.compressed_text) // 4,
                compression_ratio=cached.compression_ratio or 0.5,
                cache_hit=True,
                model_version=MODEL_VERSION,
            )

    # Determine file type
    file_type = "unknown"
    if request.file_path:
        file_type = detect_file_type(request.file_path)

    # Try to compress with models
    symbol_index = ""
    try:
        embedder = await Models.get_embedder()

        # Summarizer is optional (fallback to truncation)
        summarizer = None
        try:
            model, tokenizer = await Models.get_summarizer()
            summarizer = SummarizerWrapper(model, tokenizer)
        except Exception as e:
            logger.warning(f"Summarizer not available, using fallback: {e}")

        compressed, symbol_index = compress_with_scope_indexed(
            text=text,
            query=query,
            budget_tokens=budget,
            embedder=embedder,
            summarizer=summarizer,
            file_type=file_type,
            file_path=request.file_path,
        )
    except Exception as e:
        logger.warning(f"SCOPE compression failed, using quick fallback: {e}")
        compressed, symbol_index = quick_compress_indexed(text, budget * 4, file_type)

    compressed_tokens = len(compressed) // 4
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

    # Store in cache (include symbol_index in compressed_text for cache)
    cache_text = f"{symbol_index}\n\n{compressed}" if symbol_index else compressed
    entry = CacheEntry(
        content_hash=text_hash,
        query_hash=query_hash,
        budget_tokens=budget,
        model_version=MODEL_VERSION,
        compressed_text=cache_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=ratio,
    )
    await db.put_compressed(entry)
    await db.record_stats(hit=False)

    return CompressResponse(
        compressed_text=compressed,
        symbol_index=symbol_index if symbol_index else None,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=ratio,
        cache_hit=False,
        model_version=MODEL_VERSION,
    )


@app.post("/compress/quick", response_model=CompressResponse)
async def compress_quick(request: CompressRequest):
    """Quick compression without models (head+tail fallback)."""
    text = request.text
    budget_chars = request.budget_tokens * 4
    original_tokens = len(text) // 4

    # Determine file type for symbol extraction
    file_type = "unknown"
    if request.file_path:
        file_type = detect_file_type(request.file_path)

    compressed, symbol_index = quick_compress_indexed(text, budget_chars, file_type)
    compressed_tokens = len(compressed) // 4

    return CompressResponse(
        compressed_text=compressed,
        symbol_index=symbol_index if symbol_index else None,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=(compressed_tokens / original_tokens if original_tokens > 0 else 1.0),
        cache_hit=False,
        model_version="quick",
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize content."""
    try:
        model, tokenizer = await Models.get_summarizer()
        summarizer = SummarizerWrapper(model, tokenizer)
        summary = summarizer.summarize(request.text, request.max_tokens)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}") from e

    return SummarizeResponse(
        summary=summary,
        original_tokens=len(request.text) // 4,
        summary_tokens=len(summary) // 4,
    )


@app.post("/file-summary", response_model=FileSummaryResponse)
async def file_summary(request: FileSummaryRequest):
    """Get or create a file summary."""
    # Check cache
    cached = await db.get_file_summary(request.file_path, request.file_hash, MODEL_VERSION)
    if cached:
        return FileSummaryResponse(
            summary=cached.summary,
            summary_tokens=cached.summary_tokens or 0,
            cache_hit=True,
        )

    # Read file content if not provided
    content = request.content
    if content is None:
        try:
            with open(request.file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read file: {e}") from e

    # Generate summary
    try:
        model, tokenizer = await Models.get_summarizer()
        summarizer = SummarizerWrapper(model, tokenizer)
        summary = summarizer.summarize(content[:8000], max_length=200)
    except Exception as e:
        logger.warning(f"Summarizer failed, using quick summary: {e}")
        summary = quick_compress(content, 800)

    summary_tokens = len(summary) // 4

    # Cache it
    await db.put_file_summary(
        FileSummary(
            file_path=request.file_path,
            file_hash=request.file_hash,
            summary=summary,
            summary_tokens=summary_tokens,
            file_size=len(content),
            file_type=detect_file_type(request.file_path),
            model_version=MODEL_VERSION,
        )
    )

    return FileSummaryResponse(
        summary=summary,
        summary_tokens=summary_tokens,
        cache_hit=False,
    )


@app.post("/cache/cleanup")
async def cleanup_cache(days: int = 30):
    """Clean up old cache entries."""
    deleted = await db.cleanup_old_entries(days)
    return {"deleted": deleted}


def main():
    """Run the daemon."""
    reload_mode = "--reload" in sys.argv
    uvicorn.run(
        "scopepack.daemon:app",
        host=HOST,
        port=PORT,
        reload=reload_mode,
        log_level="info",
    )


if __name__ == "__main__":
    main()
