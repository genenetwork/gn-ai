import json
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


def init_chroma_db(docs: list, embed_model: Any, chroma_db_path: str, chunk_size: int = 1):
    if not Path(path).exists():
        raise FileNotFoundError("corpus_path is not a valid path")
    db = Chroma(
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


def create_ensemble_retriever(chroma_db: Any, docs: list, keyword_weight: float = 0.5):
    return EnsembleRetriever(
        retrievers=[
            # might need fine-tuning
            chroma_db.as_retriever(search_kwargs={"k": 20}),
            BM25Retriever.from_texts(
                texts=docs,
                metadatas=[{"source": f"Document {ind + 1}"} for ind in range(len(docs))],
                k=20,  # might need finetuning
            ),
        ],
        weights=[1 - keyword_weight, keyword_weight],
    )

