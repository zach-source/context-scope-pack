"""Embedder factory for benchmarks - supports multiple embedding models.

Configure via SCOPE_EMBEDDER env var:
- bge-small-en-v1.5 (default, local)
- cohere-embed-v4:1024 (Bedrock)
- titan-embed-text-v2:1024 (Bedrock)
"""

import os

from scopepack.embedders import create_embedder as _create_embedder
from scopepack.scope import Embedder

# Read embedder config from environment
EMBEDDER_TYPE = os.environ.get("SCOPE_EMBEDDER", "bge-small-en-v1.5")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE")

# Class-level cache to avoid reloading
_cached_embedder = None
_cached_embedder_type = None


def create_embedder(_use_real_models: bool = True) -> Embedder:
    """Create an embedder for benchmarks.

    Uses SCOPE_EMBEDDER env var to select model:
    - bge-small-en-v1.5 (default, local)
    - cohere-embed-v4:1024 (Bedrock)
    - titan-embed-text-v2:1024 (Bedrock)

    Args:
        use_real_models: Ignored - always uses real models.

    Returns:
        Configured embedder instance
    """
    global _cached_embedder, _cached_embedder_type

    if _cached_embedder is None or _cached_embedder_type != EMBEDDER_TYPE:
        print(f"Loading embedding model: {EMBEDDER_TYPE}...")
        _cached_embedder = _create_embedder(
            EMBEDDER_TYPE,
            region=AWS_REGION,
            profile=AWS_PROFILE,
        )
        _cached_embedder_type = EMBEDDER_TYPE
        print(f"Model loaded: {EMBEDDER_TYPE} ({_cached_embedder.dimensions} dims)")

    return _cached_embedder
