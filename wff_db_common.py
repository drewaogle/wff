# wf_db_common - common database operations

import weaviate
import os
import uuid
import struct
import hashlib
from typing import List
from dataclasses import dataclass

BATCH_SIZE = 100
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

@dataclass
class WFFEmbedding:
    embedding:List[float]
    img_name:str
    model_name:str
    l2:bool = False


def insert_embeddings( class_name, embeddings:List[WFFEmbedding],client=None):
    if client is None:
        client = get_client()
    with client.batch as batcher:
        batcher.batch_size = BATCH_SIZE
        batcher.timeout_retries = 3
        for e in embeddings:
            embedding_bytes = struct.pack(f'{len(e.embedding)}d', *e.embedding)
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

            # Check if embedding already exists
            query = (
                client.query.get(class_name, ["embedding_hash"])
                .with_where(
                    {
                        "path": ["embedding_hash"],
                        "operator": "Equal",
                        "valueText": embedding_hash,
                    }
                )
                .with_limit(1)
                .do()
            )
            existing = query.get("data", {}).get("Get", {}).get(class_name, [])
            if existing:
                print( f"Embedding with hash {embedding_hash} already exists in {class_name}.")
                continue

            uid = str(uuid.uuid4())
            properties = {
                "img_name": str(e.img_name),
                "model_name": e.model_name,
                "l2_normalized": e.l2,
                "embedding": e.embedding,
                "embedding_hash": embedding_hash,
            }

            batcher.add_data_object(properties, class_name, vector=e.embedding, uuid=uid)
    return True

@dataclass
class WFFSimilarEmbedding:
    img_name:str
    embedding:List[float]
    embedding_hash:str
    uid:str
    distance:float

def find_similar_embedding( class_name,
        embedding:List[float], limit=5, client=None)->List[WFFSimilarEmbedding]:
    if client is None:
        client = get_client()
    query = client.query.get(class_name, ["img_name", "embedding", "embedding_hash"])
    query = (
        query.with_near_vector({"vector": embedding})
        .with_limit(limit)
        .with_additional(["id", "distance"])
    )
    results = query.do()

    data = results.get("data", {}).get("Get", {}).get(class_name, [])

    return [
            WFFSimilarEmbedding(
                img_name = d["img_name"],
                embedding = d["embedding"],
                embedding_hash = d["embedding_hash"],
                uid = d["_additional"]["id"],
                distance = d["_additional"]["distance"]
                )
             for d in data ]

