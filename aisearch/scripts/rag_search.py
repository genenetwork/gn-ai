"""This is the main module of the package"""

import argparse
import asyncio
import os
import warnings

import dspy
import torch
from gnais.search.rag import AISearch
from gnais.search.corpus import get_docs, init_chroma_db, get_chroma_db, create_ensemble_retriever
from gnais.search.classification import extract_keywords, classify_search


warnings.filterwarnings("ignore")

CORPUS_PATH = os.getenv("CORPUS_PATH")
if CORPUS_PATH is None:
    raise FileNotFoundError("CORPUS_PATH must be specified to find corpus")
PCORPUS_PATH = os.getenv("PCORPUS_PATH")
if PCORPUS_PATH is None:
    raise FileNotFoundError("PCORPUS_PATH must be specified to read corpus")
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


dspy.configure(lm=llm, adapter=dspy.JSONAdapter())


def search(query: str, stream: bool = False):
    task_type = classify_search(query)
    chroma_db = get_chroma_db(chroma_db_path=DB_PATH, embed_model="Qwen/Qwen3-Embedding-0.6B")
    search = AISearch(
        stream=stream,
        ensemble_retriever=None
    )
    docs = get_docs(CORPUS_PATH)
    if task_type.get("decision") == "keyword":
        print("\nSettled on keyword-ish search!")
        query = extract_keywords(query)
        query = query.get("keywords")
        search.ensemble_retriever = create_ensemble_retriever(chroma_db=chroma_db, docs=docs, keyword_weight=0.7)
    else:
        search.ensemble_retriever = create_ensemble_retriever(chroma_db=chroma_db, docs=docs)
    return query, search


async def digest(query: str, stream: bool = False):
    query, _search = search(query, stream=stream)
    result = _search.handle(query)
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
