import json
from functools import lru_cache
from typing import Any
from pathlib import Path
from tqdm import tqdm
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def get_docs(corpus_dir: str) -> dict:
    corpus_dir = Path(corpus_dir)
    metadata = []
    for corpus in corpus_dir.iterdir():
        with open(corpus, "r") as data:
            metadata.extend(json.load(data))
    return metadata


def _docs_to_tuple(docs: list) -> tuple:
    """Convert docs list to a hashable tuple for caching."""
    if not docs:
        return tuple()
    if isinstance(docs[0], str):
        return tuple(docs)
    return tuple(json.dumps(d, sort_keys=True) for d in docs)


@lru_cache(maxsize=8)
def _cached_bm25_retriever(docs_tuple: tuple, k: int):
    """Build (and cache) the BM25 retriever.  Tokenizing the corpus is
    expensive, so we only do it once per unique (docs, k) pair."""
    return BM25Retriever.from_texts(
        texts=list(docs_tuple),
        metadatas=[{"source": f"Document {ind + 1}"} for ind in range(len(docs_tuple))],
        k=k,
    )


def init_chroma_db(docs: list, embed_model: Any, chroma_db_path: str, chunk_size: int = 1):
    if not Path(path).exists():
        raise FileNotFoundError("corpus_path is not a valid path")
    db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=HuggingFaceEmbeddings(
            model_name=embed_model,
            # We have NVIDIA
            model_kwargs={"trust_remote_code": True, "device": "cuda:0"},
        ),
        client_settings=Settings(
            is_persistent=True,
            persist_directory=chroma_db_path,
            anonymized_telemetry=False,
        ),
    )
    for i in tqdm(range(0, len(docs), chunk_size)):
        chunk = docs[i : i + chunk_size]
        metadatas = [
            {"source": f"Document {ind + 1}"}
                    for ind in range(i, i + len(chunk))
        ]
        db.add_texts(
            texts=chunk,
            metadatas=metadatas,
        )
    db.persist()
    return db


def get_chroma_db(chroma_db_path: str, embed_model=Any):
    return Chroma(
        persist_directory=chroma_db_path,
        embedding_function=HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        ),
        client_settings=Settings(
            is_persistent=True,
            persist_directory=chroma_db_path,
            anonymized_telemetry=False,
        ),
    )


def create_ensemble_retriever(chroma_db: Any, docs: list, keyword_weight: float = 0.5, **kwargs):
    k = kwargs.get("k") if kwargs.get("k") else 20
    c = kwargs.get("c") if kwargs.get("c") else 60
    weights = kwargs.get("weights") if kwargs.get("weights") else [1 - keyword_weight, keyword_weight]

    bm25_retriever = _cached_bm25_retriever(_docs_to_tuple(docs), k)

    return EnsembleRetriever(
        retrievers=[
            # might need fine-tuning
            chroma_db.as_retriever(search_kwargs={"k": k}),
            bm25_retriever,
        ],
        weights=weights,
        c=c,
    )

