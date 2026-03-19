"""
Module with RAG system for AI search in GeneNetwork
Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

__all__ = (
    "AISearch",
    "THREAD",
    "extract_keywords",
    "classify_search",
)

import asyncio
import json
import os
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chromadb.config import Settings
from gnais.rag.config import classify, extract, generate, reformat
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
THREAD = uuid.uuid4().hex[:8]


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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
    memory: Any = field(init=False)
    graph: Any = field(init=False)

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

        self.memory = MemorySaver()

        graph_builder = StateGraph(State)
        graph_builder.add_node("chat", self.chat)
        graph_builder.add_edge(START, "chat")
        graph_builder.add_edge("chat", END)
        self.graph = graph_builder.compile(checkpointer=self.memory)

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
            raise FileNotFoundError("corpus_path is not a valid path")

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

    def chat(self, state: State) -> dict:
        """Run user query through the RAG system for search

        Args:
            query: user query

        Returns:
            response to user query
        """
        query = state.get("messages")
        chat_history = deepcopy(query)
        new_query = deepcopy(query[-1].content)
        retrieved_docs = self.ensemble_retriever.invoke(new_query)
        system_prompt = """
            You excel at addressing search query using the context and chat history you have. You do not make mistakes.
            Extract answers to the query below from the context and chat history. Use the chat history before moving to the context.
            Provide links associated with each RDF entity. To build links you must replace RDF prefixes by namespaces.
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
            Original trait link: https://rdf.genenetwork.org/v1/id/trait_16339
            Trait id: 16339
            Dataset name: BXDPublish
            New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish\n
            """
        final_prompt = system_prompt + new_query
        response = generate(
            input_text=final_prompt,
            chat_history=chat_history,
            context=retrieved_docs,
        )
        response = str(response.get("feedback"))
        return {"messages": [response]}

    async def call_graph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": THREAD}}
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    def handle(self, query: str) -> str:
        result = asyncio.run(self.call_graph(query))
        result = result.get("messages")[-1].content
        reformatted = reformat(input_text=result).get(
            "result"
        )  # transform to valid nested dictionary
        return reformatted


def extract_keywords(query: str) -> str:
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


def classify_search(query: str) -> str:
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
