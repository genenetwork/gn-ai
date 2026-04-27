"""This is the main module of the package"""

import argparse
import asyncio
import os
import warnings

import dspy
import torch
from typing import Any
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from gnais.search.rag import rag_search
from gnais.search.corpus import get_docs, init_chroma_db, get_chroma_db, create_ensemble_retriever
from gnais.search.classification import extract_keywords, classify_search


warnings.filterwarnings("ignore")

CORPUS_PATH = os.getenv("CORPUS_PATH")
if CORPUS_PATH is None:
    raise FileNotFoundError("CORPUS_PATH must be specified to find corpus")
MEMORY_PATH = os.getenv("DB_PATH")
if MEMORY_PATH is None:
    raise FileNotFoundError("DB_PATH must be specified to access database")
DB_PATH = os.getenv("DB_PATH")
if DB_PATH is None:
    raise FileNotFoundError("DB_PATH must be specified to access database")
SEED = os.getenv("SEED")
if SEED is None:
    raise ValueError("SEED must be specified for reproducibility")
MODEL_NAME = os.getenv("MODEL_NAME")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME must be specified - either proprietary or local")
MODEL_TYPE = os.getenv("MODEL_TYPE")
if MODEL_TYPE is None:
    raise ValueError("MODEL_TYPE must be specified")


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


if int(MODEL_TYPE) == 0:
    llm = dspy.LM(
        model=f"openai/{MODEL_NAME}",
        api_base="http://localhost:7501/v1",
        api_key="local",
        model_type="chat",
        max_tokens=10_000,
        n_ctx=10_000,
        seed=2_025,
        temperature=0,
        verbose=False,
    )
elif int(MODEL_TYPE) == 1:
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Valid API_KEY must be specified to use the proprietary model")
    llm = dspy.LM(
        MODEL_NAME,
        api_key=API_KEY,
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
else:
    raise ValueError("MODEL_TYPE must be 0 or 1")


dspy.configure(lm=llm)


def digest(query: str, memory: Any, user_id: str = "default_user") -> str:
    """Run the full RAG pipeline for a single query and return the answer.

    This is a convenience wrapper around the same logic used by the CLI
    script, packaged as a plain synchronous function.
    """

    async def _run() -> str:

        docs = get_docs(CORPUS_PATH)
        chroma_db = get_chroma_db(chroma_db_path=DB_PATH, embed_model="Qwen/Qwen3-Embedding-0.6B")
        decision = classify_search(query).get("decision")
        retriever = create_ensemble_retriever(
            chroma_db=chroma_db,
            docs=docs,
            keyword_weight=0.7 if decision == "keyword" else 0.5,
        )

        parts = []
        async for chunk in rag_search(
            query=query,
            retriever=retriever,
            memory=memory,
            user_id=user_id,
            chat_history=[],
            memory_type="rag",
        ):
            parts.append(str(chunk))
        return "".join(parts)

    return asyncio.run(_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--user-id", default="test-user", help="Mem0 user identity")
    args = parser.parse_args()
    memory_config = MemoryConfig(
        llm={
            "provider": "litellm",
            "config": {
                "model": "moonshot/kimi-k2-0711-preview",
                "temperature": 0.5,
                "max_tokens": 2_000,
                "api_key": API_KEY,
            },
        },
        embedder={
            "provider": "huggingface",
            "config": {
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "embedding_dims": 1024,
                "model_kwargs": {
                    "trust_remote_code": True,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
            },
        },
        vector_store={
            "provider": "chroma",
            "config": {
                "collection_name": "mem0",
                "path": os.path.join(MEMORY_PATH, "rag_mem0_chroma"),
            },
        },
    )
    memory = Memory(config=memory_config)
    print(digest(args.query, user_id=args.user_id, memory=memory))
