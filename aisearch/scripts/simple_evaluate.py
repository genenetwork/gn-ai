"""Script for performance evaluation of GN AI systems using mere averaging"""

import os
import time
from typing import Any

import dspy
import numpy as np
import pandas as pd
import torch
from gnais.config import Config
from gnais.evaluation.utils import (
    agent_digest,
    get_dataset,
    graph_rag_digest,
    hybrid_digest,
    mark,
    rag_digest,
)


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
    precisions, recalls, f1s, speeds, n_tokens = [], [], [], [], []
    for example in evaluation_set:
        query = example.get("query")
        true_response = example.get("answer")

        start = time.time()
        result = runner(query=query)
        end = time.time()
        generated_response = result.get("answer")
        n_token = list(result.get_lm_usage().values())[0]["total_tokens"]

        with dspy.context(lm=judge_llm):
            precision, recall, f1 = mark(query, generated_response, true_response)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        speeds.append(end - start)
        n_tokens.append(n_token)

    metrics = {
        "precision": np.mean(precisions).item(),
        "recall": np.mean(recalls).item(),
        "f1": np.mean(f1s).item(),
        "speed": np.mean(speeds).item(),
        "n_token": np.mean(n_tokens).item(),
    }
    return metrics


if __name__ == "__main__":
    DATASET_PATH = os.getenv("DATASET_PATH")
    if DATASET_PATH is None:
        raise FileNotFoundError("DATASET_PATH must be set for evaluation")

    OUTPUT_PATH = os.getenv("OUTPUT_PATH")
    if OUTPUT_PATH is None:
        raise FileNotFoundError("OUTPUT_PATH must be set for evaluation")

    N_ITERATIONS = int(os.getenv("N_ITERATIONS"))
    if N_ITERATIONS is None:
        raise ValueError("N_ITERATIONS must be set for evaluation")

    JUDGE_MODEL = os.getenv("JUDGE_MODEL")
    if JUDGE_MODEL is None:
        raise ValueError("JUDGE MODEL must be set for evaluation")

    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    dspy.configure(lm=Config.DEFAULT_LLM, track_usage=True)

    judge_llm = dspy.LM(
        model=JUDGE_MODEL,
        api_key=Config.API_KEY,
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
