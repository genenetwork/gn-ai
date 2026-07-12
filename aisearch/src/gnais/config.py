__all__ = ("Config",)
import os

import dspy
from dotenv import load_dotenv

load_dotenv()


class Config:
    CORPUS_PATH = os.environ.get("CORPUS_PATH")
    if CORPUS_PATH is None:
        raise RuntimeError("CORPUS_PATH is not set")

    DB_PATH = os.environ.get("DB_PATH")
    if DB_PATH is None:
        raise RuntimeError("DB_PATH is not set")

    MEM0_PATH = os.path.join(DB_PATH, "mem0_chroma")

    SEED = int(os.environ.get("SEED"))
    if SEED is None:
        raise RuntimeError("SEED is not set")

    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")
    if DEFAULT_MODEL is None:
        raise RuntimeError("DEFAULT_MODEL is not set")

    ALTERNATIVE_MODEL = os.environ.get("ALTERNATIVE_MODEL")
    if ALTERNATIVE_MODEL is None:
        raise RuntimeError("ALTERNATIVE_MODEL is not set")

    MEMORY_MODEL = os.environ.get("MEMORY_MODEL")
    if MEMORY_MODEL is None:
        raise RuntimeError("MEMORY_MODEL is not set")

    MODEL_TYPE = int(os.environ.get("MODEL_TYPE"))
    if MODEL_TYPE is None:
        raise RuntimeError("MODEL_TYPE is not set")
    if MODEL_TYPE not in (
        1,
        2,
        3,
    ):
        raise ValueError("MODEL_TYPE must be 1, 2 or 3")

    API_KEY = os.environ.get("API_KEY")
    if MODEL_TYPE and API_KEY is None:
        raise RuntimeError("API_KEY is not set")

    SPARQL_ENDPOINT = os.environ.get("SPARQL_ENDPOINT")
    if SPARQL_ENDPOINT is None:
        raise RuntimeError("SPARQL_ENDPOINT is not set")

    AUTH_SERVER_URL = os.environ.get("AUTH_SERVER_URL")
    if AUTH_SERVER_URL is None:
        raise RuntimeError("AUTH_SERVER_URL is not set")

    SECRET_KEY = os.environ.get("SECRET_KEY")
    if SECRET_KEY is None:
        raise RuntimeError("SECRET_KEY is not set")

    PORT = os.environ.get("PORT")
    if PORT is None:
        raise RuntimeError("PORT for local model is not set")

    if MODEL_TYPE == 1:  # only frontier models
        DEFAULT_LLM = dspy.LM(
            model=DEFAULT_MODEL,
            api_key=API_KEY,
            max_tokens=50_000,
            temperature=1,
            cache=False,  # essential for evaluation; switch on for local deploy
            verbose=False,
        )
        ALTERNATIVE_LLM = dspy.LM(
            model=ALTERNATIVE_MODEL,
            api_key=API_KEY,
            max_tokens=50_000,
            temperature=1,
            cache=False,  # essential for evaluation; switch on for local deploy
            verbose=False,
        )
    elif MODEL_TYPE == 2:  # only local models
        DEFAULT_LLM = dspy.LM(
            model=f"ollama_chat/{DEFAULT_MODEL}",
            api_key="local",
            api_base=f"http://localhost:{PORT}",
            max_tokens=50_000,
            temperature=1,
            cache=False,  # essential for evaluation; switch on for local deploy
            verbose=False,
        )
        ALTERNATIVE_LLM = dspy.LM(
            model=f"ollama_chat/{ALTERNATIVE_MODEL}",
            api_key="local",
            api_base=f"http://localhost:{PORT}",
            max_tokens=50_000,
            temperature=1,
            cache=False,  # essential for evaluation; switch on for local deploy
            verbose=False,
        )
    elif MODEL_TYPE == 3:  # smart combination of frontier and local models
        DEFAULT_LLM = dspy.LM(
            model=DEFAULT_MODEL,
            api_key=API_KEY,
            max_tokens=50_000,
            temperature=1,
            cache=False,  # essential for evaluation; switch on for local deploy
            verbose=False,
        )
        ALTERNATIVE_LLM = dspy.LM(
            model=f"ollama_chat/{ALTERNATIVE_MODEL}",
            api_key="local",
            api_base=f"http://localhost:{PORT}",
            max_tokens=50_000,
            temperature=1,
            cache=False,  # essential for evaluation; switch on for local deploy
            verbose=False,
        )
