__all__ = (
    "Config",
)
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    CORPUS_PATH = os.environ.get("CORPUS_PATH")
    if CORPUS_PATH is None:
        raise RuntimeError("CORPUS_PATH is not set")

    PCORPUS_PATH = os.environ.get("PCORPUS_PATH")
    if PCORPUS_PATH is None:
        raise RuntimeError("PCORPUS_PATH is not set")

    DB_PATH = os.environ.get("DB_PATH")
    if DB_PATH is None:
        raise RuntimeError("DB_PATH is not set")

    SEED = os.environ.get("SEED")
    if SEED is None:
        raise RuntimeError("SEED is not set")

    MODEL_NAME = os.environ.get("MODEL_NAME")
    if MODEL_NAME is None:
        raise RuntimeError("MODEL_NAME is not set")

    MODEL_TYPE = int(os.environ.get("MODEL_TYPE"))
    if MODEL_TYPE is None:
        raise RuntimeError("MODEL_TYPE is not set")
    if MODEL_TYPE not in (0, 1,):
        raise ValueError("MODEL_TYPE must be 0 or 1")

    API_KEY = os.environ.get("API_KEY")
    if MODEL_TYPE and API_KEY is None:
        raise RuntimeError("API_KEY is not set")
