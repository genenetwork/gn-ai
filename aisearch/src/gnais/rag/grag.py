"""Module with GraphRAG system for AI search in GeneNetwork"""

__all__ = (
    "AISearch",
    "THREAD",
)

import asyncio
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gnais.rag.config import generate_response, generate_sparql, reformat
from gnais.rag.rag import extract_keywords
from gnais.utils import fetch_schema
from langchain_community.graphs import RdfGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from SPARQLWrapper import JSON, SPARQLWrapper
from typing_extensions import Annotated, TypedDict

THREAD = uuid.uuid4().hex[:8]


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass
class AISearch:
    """
    Represent GraphRAG system that queries RDF directly using SPARQL generation for AISearch
    No text conversion or vector embeddings needed
    """

    endpoint_url: str
    llm: Any
    rdf_classes: Any = field(init=False)
    rdf_properties: Any = field(init=False)
    rdf_graph: Any = field(init=False)
    memory: Any = field(init=False)
    lang_graph: Any = field(init=False)

    def __post_init__(self):
        # Get schema information for better SPARQL generation
        self.rdf_classes, self.rdf_properties = fetch_schema(self.endpoint_url)
        # Initialize rdf graph
        self.rdf_graph = RdfGraph(
            source_file=self.endpoint_url,
            standard="owl",
        )
        # Instantiate memory
        self.memory = MemorySaver()
        # Initialize langgraph
        graph_builder = StateGraph(State)
        graph_builder.add_node("chat", self.chat)
        graph_builder.add_edge(START, "chat")
        graph_builder.add_edge("chat", END)
        self.lang_graph = graph_builder.compile(checkpointer=self.memory)

    def chat(self, state: State) -> Dict[str, Any]:
        """
        Address a natural language question
        """
        query = state.get("messages")
        chat_history = deepcopy(query)
        new_query = deepcopy(query[-1].content)
        new_query = extract_keywords(new_query)
        system_prompt = """
               You excel at addressing search query using the context and chat history you have. You do not make mistakes.
               Extract answers to the query below from the context and chat history. Use the chat history before moving to the context.
               Provide links associated with each RDF entity. To build links you must replace RDF prefixes by namespaces using the schema provided.
               All links pointing to specific traits should be translated to CD links using the trait id and the dataset name.
               Original trait link: https://rdf.genenetwork.org/v1/id/trait_16339
               Trait id: 16339
               Dataset name: BXDPublish
               New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish\n"""
        final_prompt = system_prompt + f"Query: {new_query}"
        query_result = generate_sparql(
            original_query=final_prompt,
            classes_info=self.rdf_classes,
            properties_info=self.rdf_properties,
        )
        sparql_queries = query_result.get("sparql_queries")
        try:
            sparql = SPARQLWrapper(self.endpoint_url)
            sparql.setReturnFormat(JSON)
            final_results = []
            for sparql_query in sparql_queries:
                sparql.setQuery(sparql_query)
                results = sparql.queryAndConvert()
                results = str(results["results"]["bindings"])
                final_results.append(results)
            final_results = str(final_results)
        except Exception as e:
            final_results = f"Query failed: {str(e)}"
        response = generate_response(
            original_query=final_prompt,
            sparql_results=final_results,
            chat_history=chat_history,
        )
        response = str(response.get("feedback"))
        return {"messages": [response]}

    async def call_langgraph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": THREAD}}
        result = await self.lang_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    async def handle(self, query: str) -> str:
        result = await self.call_langgraph(query)
        result = result.get("messages")[-1].content
        reformatted = reformat(input_text=result).get(
            "result"
        )  # transform to valid format
        return reformatted
