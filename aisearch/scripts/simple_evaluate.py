"""Script for performance evaluation of GN AI systems using mere averaging"""

import argparse
import asyncio
import os
import time
import uuid
from typing import Any

import dspy
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from gnais.config import Config
from gnais.evaluation.utils import (agent_digest, get_dataset,
                                    graph_rag_digest, hybrid_digest, mark,
                                    rag_digest)
from gnais.search.agent import agent_search
from gnais.search.classification import classify_search
from gnais.search.corpus import (create_ensemble_retriever, get_chroma_db,
                                 get_docs, init_chroma_db)
from gnais.search.grag import graph_rag_search
from gnais.search.rag import rag_search
from gnais.search.ragent import hybrid_search


def make_program(
    search_func: Any,
) -> dspy.predict.react.ReAct:

    search = dspy.Tool(
        name="search",
        desc="Answer the question asked by the user.",
        args={
            "query": {
                "type": "string",
                "desc": "question asked by the user",
            },
        },
        func=search_func,
    )

    class Signature(dspy.Signature):
        query: str = dspy.InputField(desc="User question")
        answer: str = dspy.OutputField(
            desc="Answer to user's query with details and explanations"
        )

    # Wrap hybrid search into a DSPy module that is compatible with DSPy evaluation
    program = dspy.ReAct(
        signature=Signature, tools=[search], max_iters=2
    )  # limit iterations

    return program


def run_eval(
    runner: Any, evaluation_set: list[dspy.Example], judge_llm: dspy.LM
) -> dict[str, float]:
    precisions, recalls, f1s, speeds = [], [], [], []
    for example in evaluation_set:
        query = example.get("query")
        true_response = example.get("answer")
        
        start = time.time()
        generated_response = runner(query=query).get("answer")
        end = time.time()
        
        with dspy.context(lm=judge_llm):
            precision, recall, f1 = mark(query, generated_response, true_response)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        speeds.append(end-start)

    metrics = {
        "precision": np.mean(precisions).item(),
        "recall": np.mean(recalls).item(),
        "f1": np.mean(f1s).item(),
        "speed": np.mean(speeds).item(),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)

    DATASET_PATH = os.getenv("DATASET_PATH")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH")
    SEED = int(os.getenv("SEED"))
    MODEL_NAME = os.getenv("MODEL_NAME")
    MODEL_TYPE = int(os.getenv("MODEL_TYPE"))
    N_ITERATIONS = int(os.getenv("N_ITERATIONS"))
    API_KEY = os.getenv("API_KEY")
    PORT = os.getenv("PORT")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    llm = dspy.LM(
        model=MODEL_NAME if MODEL_TYPE else f"ollama_chat/{MODEL_NAME}",
        api_key=API_KEY if MODEL_TYPE else "local",
        api_base=None if MODEL_TYPE else f"http://localhost:{PORT}",
        max_tokens=10_000,
        temperature=1,
        cache=False,
        verbose=False,
    )
    dspy.configure(lm=llm)

    judge_llm = dspy.LM(
        model=JUDGE_MODEL,
        api_key=API_KEY,
        max_tokens=1_000,
        temperature=1,
        cache=False,
        verbose=False,
    )

    evaluation_set = get_dataset(DATASET_PATH)
    collection = {}

    # Run evaluation set with LLM only
    base = dspy.ChainOfThought("query -> answer: str")
    for n in range(N_ITERATIONS):
        base_metrics = run_eval(base, evaluation_set, judge_llm)
        collection[f"base_{n}"] = base_metrics

    # Run evaluation set with GN systems
    for system in [rag_digest, graph_rag_digest, agent_digest, hybrid_digest]:
        system_name = " ".join(system.__name__.split("_")[:-1])
        print(f"Running evaluation for {system_name}")
        for n in range(N_ITERATIONS):
            print(f"Iteration {n+1}")
            system_metrics = run_eval(make_program(system), evaluation_set, judge_llm)
            collection[f"{system_name} {n}"] = system_metrics
        print(f"Evaluation completed for {system_name}")
    final = pd.DataFrame(collection)
    final.to_csv(OUTPUT_PATH)
