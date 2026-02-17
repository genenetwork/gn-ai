"""
This module sets up configurations to run agent
It provides different constructs to interact with the LLM
Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

import logging
import os
import warnings
from typing import Any, Literal

import dspy
import torch
from Bio import Entrez
from Bio.Entrez import efetch, esearch, esummary, read
from langchain_core.messages import BaseMessage

CORPUS_PATH = os.getenv("CORPUS_PATH")
if CORPUS_PATH is None:
    raise ValueError("CORPUS_PATH must be specified to find corpus")

PCORPUS_PATH = os.getenv("PCORPUS_PATH")
if PCORPUS_PATH is None:
    raise ValueError("PCORPUS_PATH must be specified to read corpus")

DB_PATH = os.getenv("DB_PATH")
if DB_PATH is None:
    raise ValueError("DB_PATH must be specified to access database")

EXT_DB_PATH = os.getenv("EXT_DB_PATH")
if EXT_DB_PATH is None:
    raise ValueError("EXT_DB_PATH must be specified to save new data")

QUERY = os.getenv("QUERY")
if QUERY is None:
    raise ValueError("QUERY must be specified for program to run")

SEED = os.getenv("SEED")
if SEED is None:
    raise ValueError("SEED must be specified for reproducibility")

EMAIL = os.getenv("EMAIL")
if EMAIL is None:
    raise ValueError("EMAIL must be specified for NCBI tool calling")

Entrez.email = EMAIL

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

MODEL_NAME = os.getenv("MODEL_NAME")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME must be specified - either proprietary or local"
    )

MODEL_TYPE = os.getenv("MODEL_TYPE")
if MODEL_TYPE is None:
    raise ValueError("MODEL_TYPE must be specified")

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if int(MODEL_TYPE) == 0:
    GENERATIVE_MODEL = dspy.LM(
        model=f"openai/{MODEL_NAME}",
        api_base="http://localhost:7501/v1",
        api_key="local",
        model_type="chat",
        max_tokens=10_000,
        n_ctx=30_000,
        seed=2_025,
        temperature=0,
        verbose=False,
    )
elif int(MODEL_TYPE) == 1:
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Valid API_KEY must be specified to use the proprietary model")
    GENERATIVE_MODEL = dspy.LM(
        MODEL_NAME,
        api_key=API_KEY,
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
else:
    raise ValueError("MODEL_TYPE must be 0 or 1")


dspy.configure(lm=GENERATIVE_MODEL)


# Specialized modules for researcher


class Naturalize(dspy.Signature):
    text: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Natural English sentence")


naturalize_pred = dspy.Predict(Naturalize)


class Rephrase(dspy.Signature):
    input_text: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Reformulated query")


rephrase_pred = dspy.Predict(Rephrase)


class Analyze(dspy.Signature):
    context: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    input_text: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Analysis (≤200 words)")


analyze_pred = dspy.Predict(Analyze)


class Check(dspy.Signature):
    answer: list[BaseMessage] = dspy.InputField()
    input_text: list[BaseMessage] = dspy.InputField()
    decision: str = dspy.OutputField(desc='"yes" or "no"')


check_pred = dspy.Predict(Check)


class Summarize(dspy.Signature):
    full_context: list[BaseMessage] = dspy.InputField()
    summary: str = dspy.OutputField(desc="Bullet-point summary")


summarize_pred = dspy.Predict(Summarize)


class Synthesize(dspy.Signature):
    input_text: list[BaseMessage] = dspy.InputField()
    updated_history: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final paragraph")


synthesize_pred = dspy.Predict(Synthesize)


class Subquery(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    answer: list[str] = dspy.OutputField(desc="The list of smaller tasks")


subquery = dspy.Predict(Subquery)


class Finalize(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    subqueries: list[BaseMessage] = dspy.InputField()
    answers: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final answer")


finalize_pred = dspy.Predict(Finalize)


# Specialized ReAct architecture for expert


def search_ncbi(database: str, term: str, max_results: int = 10) -> Any:
    handle = esearch(db=database, term=term, retmax=max_results)
    records = read(handle)
    handle.close()
    return records


search_ncbi = dspy.Tool(
    name="search_ncbi",
    desc="Search an NCBI database (e.g., nucleotide, protein, pubmed) for a term",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "term": {"type": "string", "desc": "Search term or query"},
        "max_results": {
            "type": "integer",
            "desc": "Max results (default 10)",
            "default": 10,
        },
    },
    func=search_ncbi,
)


def fetch_record(database: str, record_id: str, rettype: str) -> str:
    handle = efetch(db=database, id=record_id, rettype=rettype, retmode="text")
    result = handle.readline().strip()
    handle.close()
    return result


fetch_record = dspy.Tool(
    name="fetch_record",
    desc="Fetch a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
        "rettype": {"type": "string", "desc": "Return type compatible with database"},
    },
    func=fetch_record,
)


def summarize_record(database: str, record_id: str) -> Any:
    handle = esummary(db=database, id=record_id)
    result = read(handle)
    handle.close()
    return result


summarize_record = dspy.Tool(
    name="summarize_record",
    desc="Get summary on a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
    },
    func=summarize_record,
)


class ReactSig(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    solution: str = dspy.OutputField(desc="The answer to the query")


class React(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [search_ncbi, fetch_record, summarize_record]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=50,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)


class Plan(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The plan to solve the task")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


# Module to make plan
plan = dspy.Predict(Plan)


class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The new questions")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


# Module to tune reflection
tune = dspy.Predict(Tune)


class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    next_decision: Literal["researcher", "reflector", "expert", "end"] = (
        dspy.OutputField(desc="The next step to take based on instructions")
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the decision in 50 words"
    )


# Module to manage system
supervise = dspy.Predict(Decide)


class End(dspy.Signature):
    messages: list[BaseMessage] = dspy.InputField()
    feedback: str = dspy.OutputField(
        desc="Detailed and comprehensive final feedback combining AI outputs in the list of messages and linking them when necessary"
    )


# Module to wrap up
end = dspy.Predict(End)
