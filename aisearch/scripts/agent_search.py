"""This is the main module of the package using agent tool calling"""

import argparse
import asyncio
import os
from typing import Any

import dspy
import torch
from gnais.config import Config
from gnais.search.agent import agent_search
from mem0 import Memory
from mem0.configs.base import MemoryConfig


def digest(query: str, memory: Any = None, user_id: str = "default_user"):
    async def _run():
        output = ""
        async for chunk in agent_search(
            query=query,
            sparql_url=Config.SPARQL_ENDPOINT,
            memory=memory,
            user_id=user_id,
        ):
            if isinstance(chunk, dict) and "final" in chunk:
                final = chunk["final"]
                output = final
                print(final, end="", flush=True)
            else:
                output += str(chunk)
                print(chunk, end="", flush=True)
        print()
        return output

    return asyncio.run(_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--user-id", default="test-user", help="Mem0 user identity")
    args = parser.parse_args()

    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    dspy.configure(lm=Config.DEFAULT_LLM)
    os.environ[f"{Config.MEMORY_MODEL.split('/')[0].upper()}_API_KEY"] = Config.API_KEY
    memory_config = MemoryConfig(
        llm={
            "provider": "litellm",
            "config": {
                "model": Config.MEMORY_MODEL,
                "temperature": 0.1,
                "max_tokens": 2_000,
                "api_key": Config.API_KEY,
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
                "path": os.path.join(Config.DB_PATH, "mem0_chroma"),
            },
        },
    )
    memory = Memory(config=memory_config)
    output = digest(args.query, memory=memory, user_id=args.user_id)
    print(output)
