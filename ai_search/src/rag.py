"""
RAG system for AI search in GeneNetwork
This is the main module of the package
To run: `python agent.py`
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chromadb.config import Settings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from tqdm import tqdm

from config import *


@dataclass
class AISearch:
    """
    Represents RAG system used for AI search
    Encapsulates all functionalities of the system
    Takes:
         Paths of corpus and database
    Executes operations:
         Document processing
         Document embedding and database creation
         Initialization of multi-agent graph
         Initialization of subagent graph for researcher agent
         Run of query through system
    """

    corpus_path: str
    pcorpus_path: str
    db_path: str
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)

    def __post_init__(self):

        # Process or load documents
        if not Path(self.pcorpus_path).exists():  # first time readout of corpus
            self.docs = self.corpus_to_docs(self.corpus_path)
            with open(self.pcorpus_path, "w") as file:
                file.write(json.dumps(self.docs))
        else:
            with open(self.pcorpus_path) as file:
                data = file.read()
                self.docs = json.loads(data)

        # Create or get embedding database
        self.db = self.set_chroma_db(
            docs=self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"trust_remote_code": True, "device": "cpu"},
            ),  # could use gpu instead of cpu with more RAM
            db_path=self.db_path,
        )

        # Init the ensemble retriever
        metadatas = [{"source": f"Document {ind + 1}"} for ind in range(len(self.docs))]
        bm25_retriever = BM25Retriever.from_texts(
            texts=self.docs,
            metadatas=metadatas,
            k=10,  # might need finetuning
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.db.as_retriever(
                    search_kwargs={"k": 10}
                ),  # might need finetuning
                bm25_retriever,
            ],
            weights=[0.5, 0.5],  # might need finetuning
        )


    def corpus_to_docs(
        self,
        corpus_path: str,
        chunk_size: int = 1,  # small chunk size to match embedding chunk
    ) -> list:
        """Extracts documents from file and performs processing

        Args:
            corpus_path: path to directory containing corpus
            chunk_size: minimal number of documents by iteration

        Returns:
            processed document chunks
        """

        if not Path(corpus_path).exists():
            raise ValueError("corpus_path is not a valid path")

        # Read documents from a single file in corpus path
        with open(corpus_path) as f:
            aggregated = f.read()
            collection = json.loads(aggregated)  # dictionary with key being RDF subject

        docs = []
        for key in tqdm(collection):
            concat = ""
            for value in collection[key]:
                text = f"{key} is/has {value}. "
                concat += text
            docs.append(concat)

        return docs

    def set_chroma_db(
        self, docs: list, embed_model: Any, db_path: str, chunk_size: int = 1
    ) -> Any:  # small chunk_size for atomicity and memory management
        """Initializes or reads embedding database

        Args:
            docs: processed document chunks
            embed_model: model for embedding
            db_path: path to database
            chunk_size: number of chunks to process by iteration

        Returns:
            database object for embedding
        """

        settings = Settings(
            is_persistent=True,
            persist_directory=db_path,
            anonymized_telemetry=False,
        )

        if Path(db_path).exists():
            db = Chroma(
                persist_directory=db_path,
                embedding_function=embed_model,
                client_settings=settings,
            )
            return db
        else:
            db = Chroma(
                persist_directory=db_path,
                embedding_function=embed_model,
                client_settings=settings,
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

        def answer(question: str) -> str:
            retrieved_docs = self.ensemble_retriever.invoke(question)
            system_prompt = """
            You are an expert in biology and genomics. You excel at leveraging the data or context you have been given to address any user query.
            Give an accurate and elaborate response to the query below.
            In addition, provide links that the users can visit to verify information or dig deeper. To build link you must replace RDF prefixes by namespaces.

            Below is the mapping of prefixes and namespaces:
            gn => http://rdf.genenetwork.org/v1/id
            gnc => http://rdf.genenetwork.org/v1/category
            owl => http://www.w3.org/2002/07/owl
            gnt => http://rdf.genenetwork.org/v1/term
            skos = http://www.w3.org/2004/02/skos/core
            xkos => http://rdf-vocabulary.ddialliance.org/xkos
            rdf => http://www.w3.org/1999/02/22-rdf-syntax-ns
            rdfs => http://www.w3.org/2000/01/rdf-schema
            taxon => http://purl.uniprot.org/taxonomy
            dcat => http://www.w3.org/ns/dcat
            dct => http://purl.org/dc/terms
            xsd => http://www.w3.org/2001/XMLSchema
            sdmx-measure => http://purl.org/linked-data/sdmx/2009/measure
            qb => http://purl.org/linked-data/cube
            pubmed => http://rdf.ncbi.nlm.nih.gov/pubmed
            v => http://www.w3.org/2006/vcard/ns
            foaf => http://xmlns.com/foaf/0.1
            geoSeries => http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc

            Do not make any mistakes.
            """

            response = generate(
                input_text=system_prompt,
                context=retrieved_docs,
            )
            return response.get("answer")



def main(query: str) -> str:
    search_task = AISearch(
        corpus_path=CORPUS_PATH,
        pcorpus_path=PCORPUS_PATH,
        db_path=DB_PATH,
    )
    output = search_task.answer(query)
    return output

if __name__ == "__main__":
    print(main(QUERY))
