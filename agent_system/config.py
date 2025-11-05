"""
This module sets up configurations to run agent
It provides different constructs to interact with the LLM
Embedding model = Qwen/Qwen3-Embedding-0.6B
Generative model = Qwen/Qwen2.5-7B-Instruct
Note: Need to customize paths
"""

import logging
import warnings
from typing import Literal

import dspy
from langchain_core.messages import BaseMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(
    filename="log_agent.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

# TODO: Customize path
CORPUS_PATH = "/home/johannesm/all_corpus/"

# TODO: Customize path
PCORPUS_PATH = "/home/johannesm/all_tmp/new_docs.txt"

# TODO: Customize path
DB_PATH = "/home/johannesm/all_tmp/new_chroma_db"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

GENERATIVE_MODEL = dspy.LM(
    model="openai/Qwen/Qwen2.5-7B-Instruct",  # should match shell config
    api_base="http://localhost:7501/v1",
    api_key="local",
    model_type="chat",
    max_tokens=10_000,
    n_ctx=30_000,
    seed=2_025,
    temperature=0,
    verbose=False,
)

dspy.configure(lm=GENERATIVE_MODEL)


# Main function to interact with LLM
generate = dspy.ChainOfThought("question -> answer: str")


class Subquery(dspy.Signature):
    query: str = dspy.InputField(desc="the task to solve")
    answer: list = dspy.OutputField(
        desc="the smaller tasks that help solve the main task"
    )
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )


# Specialized LLM function to extract subqueries
subquery = dspy.Predict(Subquery)


class Plan(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField(
        desc="the background to use to decide"
    )
    answer: str = dspy.OutputField(desc="the task result")
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )


# Specialized LLM function to make plan
plan = dspy.Predict(Plan)


class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField(
        desc="the background to use to ask new questions"
    )
    answer: str = dspy.OutputField(desc="the new questions")
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )


# Specialized LLM function to tune reflection
tune = dspy.Predict(Tune)


class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField(
        desc="the background to use to make decision"
    )
    next: Literal["researcher", "reflector", "end"] = dspy.OutputField(
        desc="the next step to take"
    )
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the decision given the input, limited to approximately 50 words."
    )


# Specialized LLM function to manage system
supervise = dspy.Predict(Decide)
