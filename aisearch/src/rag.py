"""
Module with RAG system for AI search in GeneNetwork
Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chromadb.config import Settings
from gnais.config import *
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class AISearch:
    """
    Represent RAG system used for AI search
    Encapsulate all functionalities of the system
    Take:
         Paths of corpus and database
    Execute operations:
         Document processing
         Document embedding and database creation
         Run of query
    """

    corpus_path: str
    pcorpus_path: str
    db_path: str
    keyword_weight: float = 0.5
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
            k=20,  # might need finetuning
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.db.as_retriever(search_kwargs={"k": 20}),  # might need finetuning
                bm25_retriever,
            ],
            weights=[1 - self.keyword_weight, self.keyword_weight],
        )

    def corpus_to_docs(
        self,
        corpus_path: str,
        chunk_size: int = 1,  # small chunk size to match embedding chunk
    ) -> list:
        """Extract documents from file and performs processing

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
        """Initialize or read embedding database

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

    def extract_keywords(self, query: str) -> str:
        """Extract list of keywords from query

        Args:
            query: user query

        Returns:
            list of keywords
        """

        system_prompt = """
            You are extremely good at extracting keywords from a search query related to specific entities (traits, markers, etc) in GeneNetwork.
            Produce a list of space separated keywords featured in the query below. Only return that list.
            \n
            """
        final_prompt = system_prompt + query
        keywords = extract(input_text=final_prompt)
        return keywords

    def classify_search(self, query: str) -> str:
        """Classify user query as keyword search or semantic search

        Args:
            query: user query

        Returns:
            type of search for query processing
        """

        system_prompt = """
            You are an experienced search classifier.
            You can accurately tell from a query if a keyword search or semantic search is more appropriate to provide satisfactory answers to the user.
            A keyword search is appropriate when specific entities feature in the query (i.e trait id, marker code, etc.).
            A semantic search is better when the system needs to understand the meaning of the query and make implicit connections.
            Infer the type of search that should be performed given the query below:
            \n
            """
        final_prompt = system_prompt + query
        search_type = classify(input_text=final_prompt)

        return search_type

    def handle(self, query: str) -> str:
        """Run user query through the RAG system for search

        Args:
            query: user query

        Returns:
            response to user query
        """
        retrieved_docs = self.ensemble_retriever.invoke(query)
        system_prompt = """
            You excel at addressing search query using the context you have. You do not make mistakes.
            Extract answers to the query from the context and provide links associated with each RDF entity.
            To build links you must replace RDF prefixes by namespaces.
            Here is the mapping of prefixes and namespaces:
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

            All links pointing to specific traits should be translated to CD links using the trait id and the dataset name.
            Original trait link: https://rdf.genenetwork.org/v1/id/trait_BXDPublish_16339
            Trait id: 16339
            Dataset name: BXDPublish
            New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish
            \n
            """
        final_prompt = system_prompt + query
        response = generate(
            input_text=final_prompt,
            context=retrieved_docs,
        )
        return response.get("feedback")
