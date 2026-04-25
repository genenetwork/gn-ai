"""Module with GraphRAG system for AI search in GeneNetwork"""

__all__ = ("AISearch",)

import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict

import dspy
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from SPARQLWrapper import JSON, SPARQLWrapper
from typing_extensions import Annotated, TypedDict

from gnais.rag.config import (
    generate_response,
    generate_response_stream,
    generate_sparql,
    reformat,
    reformulate,
)
from gnais.rag.rag import extract_keywords
from gnais.utils import fetch_schema


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
    stream: bool = False
    rdf_classes: Any = field(init=False)
    rdf_properties: Any = field(init=False)
    rdf_examples: Any = field(init=False)
    lang_graph: Any = field(init=False)
    stream_predict: Any = field(init=False)

    def __post_init__(self):
        # Get schema information for better SPARQL generation
        self.rdf_classes, self.rdf_properties, self.rdf_examples = fetch_schema(
            self.endpoint_url
        )
        # Initialize langgraph
        graph_builder = StateGraph(State)
        graph_builder.add_node("chat", self.chat)
        graph_builder.add_edge(START, "chat")
        graph_builder.add_edge("chat", END)
        self.lang_graph = graph_builder.compile()
        self.stream_predict = dspy.streamify(
            generate_response_stream,
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="feedback")
            ],
            include_final_prediction_in_output_stream=True,
        )

    def chat(self, state: State) -> Dict[str, Any]:
        """
        Address a natural language question
        """
        predictor = generate_response_stream if self.stream else generate_response
        kwargs = {"config": {"cache": False}} if self.stream else {}
        response = predictor(**self._state_to_generation_inputs(state), **kwargs)
        return {"messages": [self._prediction_feedback(response)]}

    def _run_sparql_queries(self, sparql_queries: list[str]) -> str:
        sparql = SPARQLWrapper(self.endpoint_url)
        sparql.setReturnFormat(JSON)
        final_results = []
        for sparql_query in sparql_queries:
            try:
                sparql.setQuery(sparql_query)
                results = sparql.queryAndConvert()
                results = str(results["results"]["bindings"])
                final_results.append(results)
                time.sleep(5)
            except:
                continue
        return str(final_results)

    def _prepare_generation_inputs(
        self, query: str, chat_history: list[BaseMessage]
    ) -> dict[str, Any]:
        keyword_prompt = f"""Using extracted keywords, build SPARQL SELECT queries to efficiently fetch relevant information and address the previous query.
        Query: {query}
        Keywords: {extract_keywords(query)}."""

        keyword_results = generate_sparql(
            original_query=keyword_prompt,
            rdf_classes=self.rdf_classes,
            rdf_properties=self.rdf_properties,
            rdf_examples=self.rdf_examples,
        )
        keyword_queries = keyword_results.get("sparql_queries")
        sparql_queries = keyword_queries

        reformulated_queries = reformulate(
            original_query=query, examples=self.rdf_examples
        ).get("reformulated_queries")
        for new_query in reformulated_queries:
            semantic_prompt = f"""Using the following query reformulated with terms in SPARQL, build better SPARQL SELECT queries to fetch relevant information from RDF.
            Query: {new_query}"""
            semantic_results = generate_sparql(
                original_query=semantic_prompt,
                rdf_classes=self.rdf_classes,
                rdf_properties=self.rdf_properties,
                rdf_examples=self.rdf_examples,
            )
            semantic_queries = semantic_results.get("sparql_queries")
            sparql_queries += semantic_queries

        return {
            "requirements": """You excel at addressing search query using information you have. You do not make mistakes.
            Extract answers to the query from the sparql results and chat history. Use the chat history before moving to the results.
            Provide links associated with each RDF entity.
            Format your entire response as valid HTML. Use tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>. Do not wrap the response in markdown code blocks.""",
            "original_query": query,
            "sparql_results": self._run_sparql_queries(sparql_queries),
            "chat_history": chat_history,
        }

    def _state_to_generation_inputs(self, state: State) -> dict[str, Any]:
        messages = state.get("messages")
        query = deepcopy(messages[-1].content)
        chat_history = deepcopy(messages)
        return self._prepare_generation_inputs(query, chat_history)

    def _prediction_feedback(self, prediction: Any) -> str:
        return str(prediction.get("feedback"))

    async def call_langgraph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
        result = await self.lang_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    async def _handle(self, query: str) -> str:
        result = await self.call_langgraph(query)
        result = result.get("messages")[-1].content
        reformatted = reformat(input_text=result).get(
            "result"
        )  # transform to valid nested dictionary
        return reformatted

    async def _handle_stream(self, query: str):
        output = self.stream_predict(
            **self._prepare_generation_inputs(
                query,
                [HumanMessage(content=query)],
            ),
            config={"cache": False},
        )
        async for value in output:
            if hasattr(value, "chunk"):
                yield value.chunk
            elif isinstance(value, dspy.Prediction):
                feedback = self._prediction_feedback(value)
                if feedback:
                    yield {"final": feedback}

    def handle(self, query: str) -> Any:
        if self.stream:
            return self._handle_stream(query)
        return self._handle(query)
