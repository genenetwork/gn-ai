"""Hybrid system of RAG and Agent for GeneNetwork's AI Search"""

import asyncio
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import dspy
from gnais.agent.search import digest as agent_digest
from gnais.rag.rag_search import digest as rag_digest
from gnais.rag.grag_search import digest as grag_digest
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

THREAD = uuid.uuid4().hex[:8]


class Synthesis(dspy.Signature):
    original_query: str = dspy.InputField()
    all_generation: list[BaseMessage] = dspy.InputField()
    final_synthesis: str = dspy.OutputField(
        desc="Final response from the system formatted as neat JSON dictionary with indent"
    )


synthesize = dspy.Predict(Synthesis)


class HybridState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass
class HybridSearch:

    def __post_init__(self):
        self.memory = MemorySaver()
        self.graph = self.initialize_graph()

    def run_node(self, state: HybridState, func: Any) -> dict:
        messages = deepcopy(state.get("messages"))
        if len(messages) <= 2:  # only for first queries
            query = messages[-1].content
        else:
            query = str(messages)  # provide access to memory for subsequent queries
        response = func(query)
        return response

    def rag(self, state: HybridState) -> dict:
        response = self.run_node(state, rag_digest)
        print(f"\nRAG run!\nTemporary result: {response}")
        return {"messages": [response]}

    def grag(self, state: HybridState) -> dict:
        response = self.run_node(state, grag_digest)
        print(f"\nGraphRAG run!\nTemporary result: {response}")
        return {"messages": [response]}

    def agent(self, state: HybridState) -> dict:
        response = self.run_node(state, agent_digest)
        print(f"\nAgent run!\nTemporary result: {response}")
        return {"messages": [response]}

    def augment(self, state: HybridState) -> dict:
        messages = deepcopy(state.get("messages"))
        original_query = messages[-3].content # always take query that was used for the most recent run
        response = synthesize(original_query=original_query, all_generation=messages)
        response = response.get("final_synthesis")
        return {"messages": [response]}

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

    def handle(self, query: str) -> str:
        result = asyncio.run(self.invoke_graph(query))
        result = result.get("messages")[-1].content
        print(f"\nRun of hybrid search completed!\nFinal result: {result}")
        return result
