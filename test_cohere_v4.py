#!/usr/bin/env python
"""Test Cohere v4 response format."""
import json

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

body = {
    "texts": ["test code function"],
    "input_type": "search_document",
    "embedding_types": ["float"],
    "output_dimension": 1024,
    "truncate": "RIGHT",
}

response = client.invoke_model(
    modelId="cohere.embed-v4:0",
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json",
)

result = json.loads(response["body"].read())
print("Keys:", list(result.keys()))
print("Embeddings type:", type(result.get("embeddings")))

if "embeddings" in result:
    emb = result["embeddings"]
    if isinstance(emb, dict):
        print("Embeddings is dict with keys:", list(emb.keys()))
        if "float" in emb:
            print(
                "Float embeddings shape:",
                len(emb["float"]),
                "x",
                len(emb["float"][0]) if emb["float"] else 0,
            )
    elif isinstance(emb, list):
        print("Embeddings is list, length:", len(emb))
        if emb and isinstance(emb[0], list):
            print("First embedding length:", len(emb[0]))
