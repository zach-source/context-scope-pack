"""Embedding model implementations for SCOPE compression.

Supports multiple embedding providers:
- Local: sentence-transformers (bge-small-en-v1.5)
- AWS Bedrock: Titan, Cohere

See: https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
"""

import json
import os
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from scopepack.scope import Embedder


class EmbedderType(Enum):
    """Available embedding model types."""

    # Local models (sentence-transformers)
    BGE_SMALL = "bge-small-en-v1.5"  # 384 dims, fast, good quality

    # AWS Bedrock - Amazon Titan
    TITAN_V2_256 = "titan-embed-text-v2:256"  # 256 dims
    TITAN_V2_512 = "titan-embed-text-v2:512"  # 512 dims
    TITAN_V2_1024 = "titan-embed-text-v2:1024"  # 1024 dims (default)

    # AWS Bedrock - Cohere v3
    COHERE_ENGLISH_V3 = "cohere-embed-english-v3"  # 1024 dims
    COHERE_MULTILINGUAL_V3 = "cohere-embed-multilingual-v3"  # 1024 dims

    # AWS Bedrock - Cohere v4 (multimodal, 128k context)
    COHERE_V4_256 = "cohere-embed-v4:256"  # 256 dims
    COHERE_V4_512 = "cohere-embed-v4:512"  # 512 dims
    COHERE_V4_1024 = "cohere-embed-v4:1024"  # 1024 dims
    COHERE_V4_1536 = "cohere-embed-v4:1536"  # 1536 dims (default)


# Model metadata
MODEL_INFO = {
    EmbedderType.BGE_SMALL: {
        "provider": "local",
        "model_id": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "max_tokens": 512,
        "description": "Fast local model, good for code",
    },
    EmbedderType.TITAN_V2_256: {
        "provider": "bedrock",
        "model_id": "amazon.titan-embed-text-v2:0",
        "dimensions": 256,
        "max_tokens": 8192,
        "description": "Amazon Titan V2, compact embeddings",
    },
    EmbedderType.TITAN_V2_512: {
        "provider": "bedrock",
        "model_id": "amazon.titan-embed-text-v2:0",
        "dimensions": 512,
        "max_tokens": 8192,
        "description": "Amazon Titan V2, balanced",
    },
    EmbedderType.TITAN_V2_1024: {
        "provider": "bedrock",
        "model_id": "amazon.titan-embed-text-v2:0",
        "dimensions": 1024,
        "max_tokens": 8192,
        "description": "Amazon Titan V2, highest quality",
    },
    EmbedderType.COHERE_ENGLISH_V3: {
        "provider": "bedrock",
        "model_id": "cohere.embed-english-v3",
        "dimensions": 1024,
        "max_tokens": 512,
        "description": "Cohere English, excellent for retrieval",
    },
    EmbedderType.COHERE_MULTILINGUAL_V3: {
        "provider": "bedrock",
        "model_id": "cohere.embed-multilingual-v3",
        "dimensions": 1024,
        "max_tokens": 512,
        "description": "Cohere Multilingual, 100+ languages",
    },
    EmbedderType.COHERE_V4_256: {
        "provider": "bedrock",
        "model_id": "cohere.embed-v4:0",
        "dimensions": 256,
        "max_tokens": 128000,
        "description": "Cohere V4 multimodal, 128k context, compact",
    },
    EmbedderType.COHERE_V4_512: {
        "provider": "bedrock",
        "model_id": "cohere.embed-v4:0",
        "dimensions": 512,
        "max_tokens": 128000,
        "description": "Cohere V4 multimodal, 128k context, balanced",
    },
    EmbedderType.COHERE_V4_1024: {
        "provider": "bedrock",
        "model_id": "cohere.embed-v4:0",
        "dimensions": 1024,
        "max_tokens": 128000,
        "description": "Cohere V4 multimodal, 128k context, high quality",
    },
    EmbedderType.COHERE_V4_1536: {
        "provider": "bedrock",
        "model_id": "cohere.embed-v4:0",
        "dimensions": 1536,
        "max_tokens": 128000,
        "description": "Cohere V4 multimodal, 128k context, highest quality",
    },
}


class BaseEmbedder(ABC, Embedder):
    """Base class for embedders."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for caching."""
        ...


class LocalEmbedder(BaseEmbedder):
    """Local embedder using sentence-transformers."""

    _model = None  # Class-level cache

    def __init__(self, model_type: EmbedderType = EmbedderType.BGE_SMALL):
        """Initialize local embedder.

        Args:
            model_type: Which local model to use
        """
        self.model_type = model_type
        self._info = MODEL_INFO[model_type]

        if LocalEmbedder._model is None:
            from sentence_transformers import SentenceTransformer

            LocalEmbedder._model = SentenceTransformer(self._info["model_id"])
        self.model = LocalEmbedder._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using sentence-transformers."""
        return self.model.encode(texts, convert_to_numpy=True)

    @property
    def dimensions(self) -> int:
        return self._info["dimensions"]

    @property
    def model_name(self) -> str:
        return self.model_type.value


class BedrockEmbedder(BaseEmbedder):
    """AWS Bedrock embedder supporting Titan and Cohere models."""

    def __init__(
        self,
        model_type: EmbedderType,
        region: str | None = None,
        profile: str | None = None,
    ):
        """Initialize Bedrock embedder.

        Args:
            model_type: Which Bedrock model to use
            region: AWS region (default: from env or us-east-1)
            profile: AWS profile name (optional)
        """
        self.model_type = model_type
        self._info = MODEL_INFO[model_type]

        if self._info["provider"] != "bedrock":
            raise ValueError(f"{model_type} is not a Bedrock model")

        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.profile = profile or os.environ.get("AWS_PROFILE")
        self._client = None

    @property
    def client(self):
        """Lazy-load Bedrock client."""
        if self._client is None:
            import boto3

            session_kwargs = {}
            if self.profile:
                session_kwargs["profile_name"] = self.profile

            session = boto3.Session(**session_kwargs)
            self._client = session.client(
                "bedrock-runtime",
                region_name=self.region,
            )
        return self._client

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using Bedrock API."""
        embeddings = []

        for text in texts:
            embedding = self._encode_single(text)
            embeddings.append(embedding)

        return np.array(embeddings)

    def _encode_single(self, text: str) -> list[float]:
        """Encode a single text."""
        model_id = self._info["model_id"]

        if model_id.startswith("amazon.titan"):
            return self._encode_titan(text)
        elif model_id.startswith("cohere"):
            return self._encode_cohere(text)
        else:
            raise ValueError(f"Unknown model: {model_id}")

    def _encode_titan(self, text: str) -> list[float]:
        """Encode using Amazon Titan."""
        body = {
            "inputText": text,
            "dimensions": self._info["dimensions"],
            "normalize": True,
        }

        response = self.client.invoke_model(
            modelId=self._info["model_id"],
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        return result["embedding"]

    def _encode_cohere(self, text: str) -> list[float]:
        """Encode using Cohere (v3 or v4)."""
        model_id = self._info["model_id"]
        is_v4 = model_id.startswith("cohere.embed-v4")

        if is_v4:
            # Cohere v4 API format
            body = {
                "texts": [text],
                "input_type": "search_document",
                "embedding_types": ["float"],
                "output_dimension": self._info["dimensions"],
                "truncate": "RIGHT",
            }
        else:
            # Cohere v3 API format
            body = {
                "texts": [text],
                "input_type": "search_document",
                "truncate": "END",
            }

        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        # v4 returns {"embeddings": {"float": [[...]]}}, v3 returns {"embeddings": [[...]]}
        if is_v4:
            return result["embeddings"]["float"][0]
        else:
            return result["embeddings"][0]

    @property
    def dimensions(self) -> int:
        return self._info["dimensions"]

    @property
    def model_name(self) -> str:
        return self.model_type.value


def create_embedder(
    model_type: EmbedderType | str = EmbedderType.BGE_SMALL,
    region: str | None = None,
    profile: str | None = None,
) -> BaseEmbedder:
    """Factory function to create embedders.

    Args:
        model_type: Embedder type (string or enum)
        region: AWS region for Bedrock models
        profile: AWS profile for Bedrock models

    Returns:
        Configured embedder instance

    Examples:
        # Local model (default)
        embedder = create_embedder()

        # Bedrock Titan
        embedder = create_embedder("titan-embed-text-v2:1024")

        # Bedrock Cohere
        embedder = create_embedder(EmbedderType.COHERE_ENGLISH_V3)
    """
    # Convert string to enum if needed
    if isinstance(model_type, str):
        try:
            model_type = EmbedderType(model_type)
        except ValueError:
            # Try matching by partial name
            for et in EmbedderType:
                if model_type.lower() in et.value.lower():
                    model_type = et
                    break
            else:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available: {[e.value for e in EmbedderType]}"
                )

    info = MODEL_INFO[model_type]

    if info["provider"] == "local":
        return LocalEmbedder(model_type)
    elif info["provider"] == "bedrock":
        return BedrockEmbedder(model_type, region=region, profile=profile)
    else:
        raise ValueError(f"Unknown provider: {info['provider']}")


def list_embedders() -> list[dict]:
    """List all available embedders with their info.

    Returns:
        List of embedder info dictionaries
    """
    result = []
    for model_type in EmbedderType:
        info = MODEL_INFO[model_type].copy()
        info["type"] = model_type.value
        result.append(info)
    return result
