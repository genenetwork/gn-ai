import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import torch
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

warnings.filterwarnings("ignore")


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


@lru_cache(maxsize=4)
def get_embed_model(model_name: str):
    """Load (and cache) the embedding model so it is only instantiated once.

    Automatically uses CUDA when a GPU is available, otherwise falls back to
    CPU.  (sentence-transformers does not accept torch_dtype.)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True, "device": device},
        encode_kwargs={"batch_size": 1024},
    )


@lru_cache(maxsize=2048)
def _cached_bm25_retriever(docs_tuple: tuple, k: int):
    """Build (and cache) the BM25 retriever.  Tokenizing the corpus is
    expensive, so we only do it once per unique (docs, k) pair."""
    return BM25Retriever.from_texts(
        texts=list(docs_tuple),
        metadatas=[{"source": f"Document {ind + 1}"} for ind in range(len(docs_tuple))],
        k=k,
    )


def init_chroma_db(
    docs: list,
    embed_model: Any,
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
    chunk_size: int = 1024,
):
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    db = Chroma(
        client=client,
        embedding_function=get_embed_model(embed_model),
    )
    for i in tqdm(range(0, len(docs), chunk_size)):
        chunk = docs[i : i + chunk_size]
        metadatas = [
            {"source": f"Document {ind + 1}"} for ind in range(i, i + len(chunk))
        ]
        db.add_texts(
            texts=chunk,
            metadatas=metadatas,
        )
    # No db.persist() needed — the server handles persistence
    return db


def get_chroma_db(
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
    embed_model: Any = None,
):
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    return Chroma(
        client=client,
        embedding_function=get_embed_model(embed_model),
    )


def create_ensemble_retriever(
    chroma_db: Any, docs: list, keyword_weight: float = 0.5, **kwargs
):
    k = kwargs.get("k") if kwargs.get("k") else 20
    c = kwargs.get("c") if kwargs.get("c") else 60
    weights = (
        kwargs.get("weights")
        if kwargs.get("weights")
        else [1 - keyword_weight, keyword_weight]
    )

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
