import os
import warnings

import asyncio
import dspy
import torch
from gnais.rag.grag import *

warnings.filterwarnings("ignore")

SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT")
if SPARQL_ENDPOINT is None:
    raise ValueError("SPARQL_ENDPOINT must be specified to access database")

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


def search(query: str):
    set_search = AISearch(endpoint_url=SPARQL_ENDPOINT, llm=llm)
    return query, set_search


async def digest(query: str):
    query, set_search = search(query)
    output = await set_search.handle(query)
    return output
