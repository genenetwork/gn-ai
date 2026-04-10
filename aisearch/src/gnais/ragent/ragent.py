"""Hybrid system of RAG and Agent for GeneNetwork's AI Search"""

__all__ = (
    "HybridSearch",
    "Synthesis",
    "synthesize",
    "HybridState",
    "THREAD",
    "SearchResult",
)

import asyncio
import uuid
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing_extensions import Annotated, TypedDict

import dspy
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from gnais.agent.search import digest as agent_digest
from gnais.rag.rag_search import digest as rag_digest
from gnais.rag.grag_search import digest as grag_digest

THREAD = uuid.uuid4().hex[:8]


class SearchResult(TypedDict):
    """Standardized search result schema with flexible results array"""
    query: str
    status: str  # "success" or "partial_success"
    summary: str
    results: list[dict]  # Each result has: type, name, description, url, extra
    note: str


class Synthesis(dspy.Signature):
    """Synthesize the final response from all search components.

    Output must be a valid JSON object with this exact structure:
    {
      "query": "the original user query",
      "status": "success" or "partial_success",
      "summary": "overall summary of findings",
      "results": [
        {
          "type": "gene" | "trait" | "dataset" | "phenotype" | "locus" | etc,
          "name": "display name of the resource",
          "description": "description of relevance to query",
          "url": "URL to the resource page",
          "extra": "optional additional context like evidence, IDs, etc."
        }
      ],
      "note": "optional note about data limitations or errors"
    }

    Guidelines for results:
    - Use consistent "type" values: "gene", "trait", "dataset", "phenotype", "locus", etc.
    - Always include URL when available
    - Put specific IDs or evidence in "extra" field
    - Group similar items together in the array
    """
    original_query: str = dspy.InputField()
    all_generation: list[BaseMessage] = dspy.InputField()
    final_synthesis: SearchResult = dspy.OutputField(
        desc="Final response as a JSON object following the SearchResult schema exactly"
    )


synthesize = dspy.Predict(Synthesis)


class HybridState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass
class HybridSearch:

    def __post_init__(self):
        self.memory = MemorySaver()
        self.graph = self.initialize_graph()

    async def run_node(self, state: HybridState, func: Any) -> dict:
        messages = deepcopy(state.get("messages"))
        if len(messages) <= 2:  # only for first queries
            query = messages[-1].content
        else:
            # provide access to memory for subsequent queries
            query = str(messages)
        response = await func(query)
        return response

    async def rag(self, state: HybridState) -> dict:
        response = await self.run_node(state, rag_digest)
        print(f"\nRAG run!\nTemporary result: {response}")
        return {"messages": [response]}

    async def grag(self, state: HybridState) -> dict:
        response = await self.run_node(state, grag_digest)
        print(f"\nGraphRAG run!\nTemporary result: {response}")
        return {"messages": [response]}

    async def agent(self, state: HybridState) -> dict:
        response = await self.run_node(state, agent_digest)
        print(f"\nAgent run!\nTemporary result: {response}")
        return {"messages": [response]}

    def augment(self, state: HybridState) -> dict:
        messages = deepcopy(state.get("messages"))
        # always take query that was used for the most recent run
        original_query = messages[-3].content
        response = synthesize(
            original_query=original_query, all_generation=messages)
        # The response is now a SearchResult dict, convert to JSON string
        result_dict = response.get("final_synthesis")
        response_json = json.dumps(result_dict, indent=2)
        return {"messages": [response_json]}

    def initialize_graph(self) -> Any:
        graph_builder = StateGraph(HybridState)
        graph_builder.add_node("rag", self.rag)
        graph_builder.add_node("grag", self.grag)
        graph_builder.add_node("agent", self.agent)
        graph_builder.add_node("augment", self.augment)
        graph_builder.add_edge(START, "rag")
        graph_builder.add_edge(START, "grag")
        graph_builder.add_edge(START, "agent")
        graph_builder.add_edge("rag", "augment")
        graph_builder.add_edge("grag", "augment")
        graph_builder.add_edge("agent", "augment")
        graph_builder.add_edge("augment", END)
        graph = graph_builder.compile(checkpointer=self.memory)
        return graph

    async def invoke_graph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": THREAD}}
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    async def handle(self, query: str) -> str:
        result = await self.invoke_graph(query)
        result = result.get("messages")[-1].content
        print(f"\nRun of hybrid search completed!\nFinal result: {result}")
        return result
