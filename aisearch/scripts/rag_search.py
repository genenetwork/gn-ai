"""This is the main module of the package"""

import argparse
import asyncio
import os
import warnings

import dspy
import torch
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from gnais.search.rag import rag_search
from gnais.search.corpus import get_docs, init_chroma_db, get_chroma_db, create_ensemble_retriever
from gnais.search.classification import extract_keywords, classify_search


warnings.filterwarnings("ignore")

CORPUS_PATH = os.getenv("CORPUS_PATH")
if CORPUS_PATH is None:
    raise FileNotFoundError("CORPUS_PATH must be specified to find corpus")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response incrementally",
    )
    args = parser.parse_args()

    memory_config = MemoryConfig(
        llm={
            "provider": "litellm",
            "config": {
                "model": "moonshot/kimi-k2-0711-preview",
                "temperature": 0.1,
                "max_tokens": 2_000,
            },
        },
        embedder={
            "provider": "huggingface",
            "config": {
                "model": "Qwen/Qwen3-Embedding-0.6B",
            },
        },
        vector_store={
            "provider": "chroma",
            "config": {
                "collection_name": "mem0",
                "path": os.path.join(DB_PATH, "rag_mem0_chroma"),
            },
        },
        history_store={
            "provider": "sqlite",
            "config": {
                "path": os.path.join(DB_PATH, "rag_mem0_history.db"),
            },
        },
    )
    memory = Memory(config=memory_config)

    async def _consume():
        async for chunk in rag_search(
                query=args.query,
                chroma_db=get_chroma_db(chroma_db_path=DB_PATH, embed_model="Qwen/Qwen3-Embedding-0.6B"),
                docs=get_docs(CORPUS_PATH),
                memory=memory,
        ):
            print(chunk, end="", flush=True)
        print()
        print("Done")

    asyncio.run(_consume())
