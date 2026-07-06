"""Script for performance evaluation of GN AI systems using DSPy.Evaluate and custom decision function"""

import argparse
import asyncio
import os
import uuid
from typing import Any

import dspy
import pandas as pd
import torch
from dotenv import load_dotenv
from gnais.config import Config
from gnais.evaluation.utils import (
    agent_digest,
    get_dataset,
    graph_rag_digest,
    hybrid_digest,
    mark,
    rag_digest,
)
from gnais.search.agent import agent_search
from gnais.search.classification import classify_search
from gnais.search.corpus import (
    create_ensemble_retriever,
    get_chroma_db,
    get_docs,
    init_chroma_db,
)
from gnais.search.grag import graph_rag_search
from gnais.search.rag import rag_search
from gnais.search.ragent import hybrid_search


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
