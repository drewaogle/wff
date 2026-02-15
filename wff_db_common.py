# wf_db_common - common database operations

import weaviate
import os

_client = None
def get_client( host:str="localhost", port:int="8080"):
    global _client
    if _client is None:
        _client = weaviate.Client( f"http://{host}:{port}")
    return _client

def has_class( class_name, client = None ) -> bool:
    if client is None:
        client = get_client()
    existing_schema = client.schema.get()
    existing_classes = {c["class"] for c in existing_schema.get("classes", [])}

    return class_name in existing_classes


def create_embedding_class( class_name, model_name, l2_normalized=False, client=None ):
    if client is None:
        client = get_client()
    res = client.schema.create_class(
    {
        "class": class_name,
        "vectorIndexType": "hnsw",
        "vectorizer": "none",
        "vectorIndexConfig": {
            "M": int(os.getenv("WEAVIATE_HNSW_M", "16")),
            "distance": "cosine" if l2_normalized else "l2-squared",
        },
        "properties": [
            {"name": "img_name", "dataType": ["text"]},
            {"name": "model_name", "dataType": ["text"]},
            {"name": "l2_normalized", "dataType": ["boolean"]},
            {"name": "embedding_hash", "dataType": ["text"]},
            # embedding property is optional since we pass it as vector
            {"name": "embedding", "dataType": ["number[]"]},
        ],
    }
    )
    return not "Errors" in res

