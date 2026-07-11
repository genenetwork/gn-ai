"""Script for performance evaluation of GN AI systems using DSPy.Evaluate and custom decision function"""

import os
from typing import Any

import dspy
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


def evaluator(
    gold: dspy.Example,
    pred: str,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> int:
    query = gold["query"]
    true_response = gold["answer"]
    precision, recall, f1 = mark(query, pred, true_response)
    return 1 if (f1 >= 0.6 or recall >= 0.7 or precision >= 0.7) else 0


def run_eval(
    runner: dspy.Evaluate,
    evaluation_set: list[dspy.Example],
    search_func: Any,
) -> dspy.evaluate.EvaluationResult:

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

    return runner(program)


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

    dspy.configure(lm=Config.DEFAULT_LLM)

    judge_llm = dspy.LM(
        model=JUDGE_MODEL,
        api_key=Config.API_KEY,
        max_tokens=1_000,
        temperature=1,
        cache=False,
        verbose=False,
    )

    evaluation_set = get_dataset(DATASET_PATH)
    evaluate = dspy.Evaluate(
        devset=evaluation_set,
        metric=evaluator,
        num_threads=1,
        provide_traceback=True,
        display_table=False,
        display_progress=True,
        lm=judge_llm,
    )
    collection = {}

    # Run evaluation set with LLM only
    base = dspy.ChainOfThought("query -> answer: str")
    base_output = [evaluate(base).get("score") for n in range(N_ITERATIONS)]
    collection["base"] = base_output

    # Run evaluation set with GN systems
    for system in [rag_digest, graph_rag_digest, agent_digest, hybrid_digest]:
        system_name = " ".join(system.__name__.split("_")[:-1])
        print(f"Running evaluation for {system_name}")
        temp = []
        for n in range(N_ITERATIONS):
            print(f"Iteration {n+1}")
            system_output = run_eval(evaluate, evaluation_set, system)
            temp.append(system_output.get("score"))
        collection[system_name] = temp
        print(f"Evaluation completed for {system_name}")
    final = pd.DataFrame(collection)
    final.to_csv(OUTPUT_PATH, index=False)
