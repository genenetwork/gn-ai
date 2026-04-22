"""This is the main module of the package using agent tool calling"""

import argparse
import asyncio
import os

import dspy
import torch
from gnais.search.agent import Digest

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


dspy.configure(lm=llm, adapter=dspy.JSONAdapter())


async def digest(query: str, stream: bool = False):
    digest = Digest(stream=stream)
    result = digest.handle(query)
    if stream:
        output = ""
        async for chunk in result:
            if isinstance(chunk, dict) and "final" in chunk:
                final = chunk["final"]
                output = final
                print(final, end="", flush=True)
            else:
                output += chunk
                print(chunk, end="", flush=True)
        print()
        return output

    output = await result
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response incrementally",
    )
    args = parser.parse_args()

    output = asyncio.run(digest(args.query, stream=args.stream))
    if not args.stream:
        print(output)
