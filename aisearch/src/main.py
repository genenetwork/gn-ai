"""
This is the main module of the package
To run: `python main.py`
"""

import os

from aisearch.rag import *


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


def main(query: str):
    general_search = AISearch(
        corpus_path=CORPUS_PATH,
        pcorpus_path=PCORPUS_PATH,
        db_path=DB_PATH,
    )
    task_type = general_search.classify_search(query)
    if task_type.get("decision") == "keyword":
        new_query = general_search.extract_keywords(query)
        new_query = new_query.get("keywords")
        # Run a targeted search
        targeted_search = AISearch(
            corpus_path=CORPUS_PATH,
            pcorpus_path=PCORPUS_PATH,
            db_path=DB_PATH,
            keyword_weight=0.7,
        )
        output = targeted_search.handle(
            new_query
        )  # use extracted keywords instead for hybrid search
    else:
        output = general_search.handle(
            query
        )  # run a general search with user query straight
    return output.model_dump_json(indent=4)


if __name__ == "__main__":
    print(main(QUERY))
