"""
This module sets up configurations for the RAG
It also provides different constructs to interact with the LLM

Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

import os

import dspy
import torch
from pydantic import BaseModel, Field

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"


CORPUS_PATH = os.getenv("CORPUS_PATH")
if CORPUS_PATH is None:
    raise ValueError("CORPUS_PATH must be specified to find corpus")
PCORPUS_PATH = os.getenv("PCORPUS_PATH")
if PCORPUS_PATH is None:
    raise ValueError("PCORPUS_PATH must be specified to read corpus")
DB_PATH = os.getenv("DB_PATH")
if DB_PATH is None:
    raise ValueError("DB_PATH must be specified to access database")
QUERY = os.getenv("QUERY")
if QUERY is None:
    raise ValueError("QUERY must be specified for program to run")
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


class Classification(dspy.Signature):
    input_text: str = dspy.InputField()
    decision: str = dspy.OutputField(desc='"keyword" or "semantic"')


classify = dspy.Predict(Classification)


class Extraction(dspy.Signature):
    input_text: str = dspy.InputField()
    keywords: str = dspy.OutputField()


extract = dspy.Predict(Extraction)


class Information(BaseModel):
    """Extract relevant information for query"""

    answer: str = Field(
        description="Specific point addressing the query from the context"
    )
    links: list[str] = Field(
        description="All links associated to RDF entities related to the point"
    )


class ListInformation(BaseModel):
    """Address recursively a query"""

    detailed_answers: list[Information] = Field(
        description="List of answers to the query"
    )
    final_answer: str = Field(
        description="Synthesized and comprehensive answer using detailed answers"
    )


class Generation(dspy.Signature):
    """Wrap generation interface"""

    context: list = dspy.InputField(desc="Background information")
    input_text: str = dspy.InputField(desc="Query and instructions")
    feedback: ListInformation = dspy.OutputField(desc="System response to the query")


generate = dspy.Predict(Generation)
