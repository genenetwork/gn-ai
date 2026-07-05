"""Script for performance evaluation of GN AI systems using DSPy.Evaluate and custom decision function"""

import argparse
import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dspy
import pandas as pd
import torch
from dotenv import load_dotenv

from gnais.config import Config
from gnais.evaluation.utils import get_dataset, mark
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


_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search")


def _run_async(async_fn, *args, **kwargs):
    """Run async function in a separate thread with a fresh event loop"""

    def _worker():
        return asyncio.run(async_fn(*args, **kwargs))

    return _TOOL_EXECUTOR.submit(_worker).result()


# Wrap search functions inside conveniences
def rag_digest(query: str, memory: Any = None) -> str:
    async def _run() -> str:
        docs = get_docs(Config.CORPUS_PATH)
        chroma_db = get_chroma_db(embed_model="Qwen/Qwen3-Embedding-0.6B")
        decision = classify_search(query).get("decision")
        retriever = create_ensemble_retriever(
            chroma_db=chroma_db,
            docs=docs,
            keyword_weight=0.7 if decision == "keyword" else 0.5,
        )

        parts = []
        async for chunk in rag_search(
            query=query,
            retriever=retriever,
            memory=memory,
            user_id=str(uuid.uuid4()),
        ):
            parts.append(str(chunk))
        return "".join(parts)

    return _run_async(_run)


def sparql_digest(
    query: str, handler: Any, memory: Any = None, user_id: str = "default_user"
) -> str:
    async def _run():
        parts = []
        async for chunk in handler(
            query=query,
            sparql_url=Config.SPARQL_ENDPOINT,
            memory=memory,
            user_id=user_id,
        ):
            parts.append(str(chunk))
        return "".join(parts)

    return _run_async(_run)


def graph_rag_digest(query: str, memory: Any = None) -> str:
    return sparql_digest(
        query, graph_rag_search, memory=memory, user_id=str(uuid.uuid4())
    )


def agent_digest(query: str, memory: Any = None) -> str:
    return sparql_digest(query, agent_search, memory=memory, user_id=str(uuid.uuid4()))


def hybrid_digest(query: str, memory: Any = None) -> str:
    async def _run():
        final_html = None
        async for event in hybrid_search(
            query, memory=memory, user_id=str(uuid.uuid4())
        ):
            source = event["source"]
            kind = event["kind"]
            content = event["content"]
            if source == "hybrid" and kind == "final":
                final_html = content
                break
        return final_html

    return _run_async(_run)


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

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

        llm = dspy.LM(
            model=MODEL_NAME if MODEL_TYPE else f"openai/{MODEL_NAME}",
            api_key=API_KEY if MODEL_TYPE else "local",
            api_base=None if MODEL_TYPE else f"http://localhost:{PORT}/v1",
            max_tokens=10_000,
            temperature=1,
            cache=False,
            verbose=False,
        )
    dspy.configure(lm=llm)

    evaluation_set = get_dataset(DATASET_PATH)
    evaluate = dspy.Evaluate(
        devset=evaluation_set,
        metric=evaluator,
        num_threads=1,
        provide_traceback=True,
        display_table=False,
        display_progress=True,
        lm=llm,
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
