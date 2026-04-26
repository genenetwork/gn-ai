"""Agent with sparql tool calling for AI search in GeneNetwork"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
from langchain_community.graphs import RdfGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from SPARQLWrapper import JSON, SPARQLWrapper
from typing_extensions import Annotated, TypedDict

from gnais.rag.config import ListInformation, reformat
from gnais.utils import fetch_schema

SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT")
if SPARQL_ENDPOINT is None:
    raise ValueError(
        "SPARQL_ENDPOINT must be specified to extract RDF schema and build queries"
    )


class QueryTranslation(dspy.Signature):
    original_query: str = dspy.InputField()
    rdf_classes: list = dspy.InputField(desc="RDF classes extracted from the graph")
    rdf_properties: list = dspy.InputField(
        desc="RDF properties extracted from the graph"
    )
    rdf_examples: list = dspy.InputField(
        desc="Real RDF examples in the graph that you can use to build correct SPARQL queries"
    )
    translated_query: str = dspy.OutputField(
        desc="SPARQL query corresponding to user query for fetching requested data given RDF schema"
    )


translate_query = dspy.Predict(QueryTranslation)


def fetch_data(query: str) -> Any:
    rdf_classes, rdf_properties, rdf_examples = fetch_schema(SPARQL_ENDPOINT)
    sparql_query = translate_query(
        original_query=query,
        rdf_classes=rdf_classes,
        rdf_properties=rdf_properties,
        rdf_examples=rdf_examples,
    ).get("translated_query")
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(sparql_query)
    return sparql.queryAndConvert()


fetch_data = dspy.Tool(
    name="fetch_data",
    desc="Fetch RDF data around GeneNetwork data through SPARQL",
    args={
        "query": {
            "type": "string",
            "desc": "SPARQL query to run to fetch relevant data",
        },
    },
    func=fetch_data,
)


class ReactSig(dspy.Signature):
    query: str = dspy.InputField()
    solution: ListInformation = dspy.OutputField(desc="The answer to the query")


class StreamReactSig(dspy.Signature):
    query: str = dspy.InputField()
    solution: str = dspy.OutputField(
        desc="The answer to the query with detailed answers and the final answer, formatted as valid HTML using tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>"
    )


class AISearch(dspy.Module):
    def __init__(self, stream: bool = False):
        super().__init__()
        self.tools = [fetch_data]
        signature = StreamReactSig if stream else ReactSig

        self.react = dspy.ReAct(
            signature=signature,
            tools=self.tools,
            max_iters=20,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: str, **kwargs):
        return self.react(query=query, **kwargs)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass
class Digest:
    stream: bool = False
    search: Any = field(init=False)
    stream_search: Any = field(init=False)
    graph: Any = field(init=False)

    def __post_init__(self):
        self.search = AISearch(stream=self.stream)
        self.stream_search = dspy.streamify(
            self.search,
            stream_listeners=[
                dspy.streaming.StreamListener(
                    signature_field_name="next_thought",
                    allow_reuse=True,
                ),
                dspy.streaming.StreamListener(signature_field_name="solution"),
            ],
            include_final_prediction_in_output_stream=True,
        )
        graph_builder = StateGraph(State)
        graph_builder.add_node("chat", self.chat)
        graph_builder.add_edge(START, "chat")
        graph_builder.add_edge("chat", END)
        self.graph = graph_builder.compile()

    def _build_query(self, query: str) -> str:
        system_prompt = """
            You excel at addressing search query using the information you have. You do not make mistakes.
            Extract answers to the query from the context.
            Provide links associated with each trait. Trait usually have an object corresponding to the actual link with the gnt:has_trait_page predicate.
            Format your entire response as valid HTML. Use tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>. Do not wrap the response in markdown code blocks.
            """
        return f"{system_prompt}\n{query}"

    def _state_to_query(self, state: State) -> str:
        messages = state.get("messages")
        return self._build_query(str(messages[-1].content))

    def _prediction_solution(self, prediction: Any) -> str:
        return str(prediction.get("solution"))

    def chat(self, state: State) -> dict:
        final_query = self._state_to_query(state)
        kwargs = {"config": {"cache": False}} if self.stream else {}
        output = self.search(query=final_query, **kwargs).get("solution")
        output = str(output)
        return {"messages": [output]}

    async def call_graph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    async def _handle(self, query: str) -> str:
        result = await self.call_graph(query)
        result = result.get("messages")[-1].content
        reformatted = reformat(input_text=result).get(
            "result"
        )  # transform to valid nested dictionary
        return reformatted

    async def _handle_stream(self, query: str):
        output = self.stream_search(
            query=self._build_query(query),
            config={"cache": False},
        )
        async for value in output:
            if hasattr(value, "chunk"):
                yield value.chunk
            elif isinstance(value, dspy.Prediction):
                solution = self._prediction_solution(value)
                if solution:
                    yield {"final": solution}

    def handle(self, query: str) -> Any:
        if self.stream:
            return self._handle_stream(query)
        return self._handle(query)
