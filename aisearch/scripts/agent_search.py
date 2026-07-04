"""This is the main module of the package using agent tool calling"""

import argparse
import asyncio
import os
from typing import Any

import dspy
import torch
from dotenv import load_dotenv
from gnais.config import Config
from gnais.search.agent import agent_search
from mem0 import Memory
from mem0.configs.base import MemoryConfig


def digest(query: str, memory: Any = None, user_id: str = "default_user"):
    async def _run():
        output = ""
        async for chunk in agent_search(
            query=query, sparql_url=Config.SPARQL_ENDPOINT, memory=memory, user_id=user_id
        ):
            if isinstance(chunk, dict) and "final" in chunk:
                final = chunk["final"]
                output = final
                print(final, end="", flush=True)
            else:
                output += chunk
                print(chunk, end="", flush=True)
        print()
        return output

    return asyncio.run(_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--user-id", default="test-user", help="Mem0 user identity")
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)

    DB_PATH = os.getenv("DB_PATH")
    SEED = int(os.getenv("SEED"))
    MODEL_NAME = os.getenv("MODEL_NAME")
    MODEL_TYPE = int(os.getenv("MODEL_TYPE"))
    API_KEY = os.getenv("API_KEY")
    PORT = os.getenv("PORT")
    
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    llm = dspy.LM(
        model=MODEL_NAME if MODEL_TYPE else f"openai/{MODEL_NAME}",
        api_key=API_KEY if MODEL_TYPE else "local",
        api_base = None if MODEL_TYPE else f"http://localhost:{PORT}/v1",
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
    dspy.configure(lm=llm)

    os.environ[f"{MODEL_NAME.split('/')[0].upper()}_API_KEY"] = API_KEY
    memory_config = MemoryConfig(
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
    output = digest(args.query, memory=memory, user_id=args.user_id)
    print(output)
