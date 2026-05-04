"""This is the main module of the package"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from functools import partial

import dspy
import torch
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from gnais.search.rag import rag_search
from gnais.search.corpus import get_docs, init_chroma_db, get_chroma_db, create_ensemble_retriever
from gnais.search.classification import extract_keywords, classify_search
from gnais.search.prompts import (
    GN_FACT_EXTRACTION_PROMPT,
    GN_UPDATE_MEMORY_PROMPT,
)

def _digest(query: str, memory: Memory, user_id: str, corpus_path: str, db_path: str) -> str:
    """Run the full RAG pipeline for a single query and return the answer.

    This is a convenience wrapper around the same logic used by the CLI
    script, packaged as a plain synchronous function.
    """

    async def _run() -> str:

        docs = get_docs(corpus_path)
        chroma_db = get_chroma_db(chroma_db_path=db_path, embed_model="Qwen/Qwen3-Embedding-0.6B")
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
        ):
            parts.append(str(chunk))
        return "".join(parts)

    return asyncio.run(_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--user-id", default="test-user", help="Mem0 user identity")
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)

    CORPUS_PATH = os.getenv("CORPUS_PATH")
    DB_PATH = os.getenv("DB_PATH")
    SEED = os.getenv("SEED")
    MODEL_NAME = os.getenv("MODEL_NAME")
    MODEL_TYPE = os.getenv("MODEL_TYPE")

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

    # NOTE: Find a better way of doing this
    # This a turnaround
    # With litellm provider in MemoryConfig, a MOONSHOT_API_KEY or ANTHROPIC_API_KEY is expected
    if "moonshot" in MODEL_NAME.lower():
        os.environ["MOONSHOT_API_KEY"]=API_KEY
    elif "anthropic" in MODEL_NAME.lower():
        os.environ["ANTHROPIC_API_KEY"]=API_KEY
    
    memory_config = MemoryConfig(
        custom_fact_extraction_prompt=GN_FACT_EXTRACTION_PROMPT,
        custom_update_extraction_prompt=GN_UPDATE_MEMORY_PROMPT,
        llm={
            "provider": "litellm",
            "config": {
                "model": MODEL_NAME,
                "temperature": 0.1,
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
                "path": os.path.join(DB_PATH, "mem0_chroma"),
            },
        },
    )
    memory = Memory(config=memory_config)
    digest = partial(
        _digest,
        memory=memory,
        corpus_path=CORPUS_PATH,
        db_path=DB_PATH
    )
    print(digest(query=args.query, user_id=args.user_id))
