"""Hybrid system of RAG and Agent for GeneNetwork's AI Search"""

__all__ = (
    "HybridSearch",
    "Synthesis",
    "synthesize",
    "HybridState",
    "SearchResult",
)

import asyncio
import uuid
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import dspy
from gnais.search.agent import Digest as AgentSearch
from gnais.config import Config
from gnais.search.grag import GraphRAGSearch
from gnais.search.rag import RAGSearch
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


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
        desc="Final response from the system formatted as neat JSON dictionary with indent"
    )


synthesize = dspy.Predict(Synthesis)


class HybridState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass
class HybridSearch:
    stream: bool = False
    rag_search: Any = field(init=False)
    grag_search: Any = field(init=False)
    agent_search: Any = field(init=False)
    rag_stream_search: Any = field(init=False)
    grag_stream_search: Any = field(init=False)
    agent_stream_search: Any = field(init=False)
    graph: Any = field(init=False)

    def __post_init__(self):
        self.rag_search = RAGSearch(
            corpus_path=Config.CORPUS_PATH,
            pcorpus_path=Config.PCORPUS_PATH,
            db_path=Config.DB_PATH,
            stream=False,
        )
        self.grag_search = GraphRAGSearch(
            endpoint_url=Config.SPARQL_ENDPOINT,
            llm=dspy.settings.lm,
            stream=False,
        )
        self.agent_search = AgentSearch(stream=False)
        self.rag_stream_search = RAGSearch(
            corpus_path=Config.CORPUS_PATH,
            pcorpus_path=Config.PCORPUS_PATH,
            db_path=Config.DB_PATH,
            stream=True,
        )
        self.grag_stream_search = GraphRAGSearch(
            endpoint_url=Config.SPARQL_ENDPOINT,
            llm=dspy.settings.lm,
            stream=True,
        )
        self.agent_stream_search = AgentSearch(stream=True)
        self.graph = self.initialize_graph()

    def _stream_event(self, source: str, kind: str, content: str = "") -> str:
        return (
            json.dumps(
                {
                    "source": source,
                    "kind": kind,
                    "content": content,
                }
            )
            + "\n"
        )

    async def _stream_component(
        self, source: str, search: Any, query: str, queue: asyncio.Queue
    ) -> None:
        try:
            async for chunk in search.handle(query):
                if isinstance(chunk, dict) and "final" in chunk:
                    await queue.put((source, "final", chunk["final"]))
                else:
                    await queue.put((source, "chunk", chunk))
        except Exception as exc:
            await queue.put((source, "error", str(exc)))
        finally:
            await queue.put((source, "done", ""))

    async def run_node(self, state: HybridState, search: Any) -> dict:
        messages = deepcopy(state.get("messages"))
        if len(messages) <= 2:  # only for first queries
            query = messages[-1].content
        else:
            # provide access to memory for subsequent queries
            query = str(messages)
        response = await search.handle(query)
        return response

    async def rag(self, state: HybridState) -> dict:
        response = await self.run_node(state, self.rag_search)
        return {"messages": [response]}

    async def grag(self, state: HybridState) -> dict:
        response = await self.run_node(state, self.grag_search)
        return {"messages": [response]}

    async def agent(self, state: HybridState) -> dict:
        response = await self.run_node(state, self.agent_search)
        return {"messages": [response]}

    def augment(self, state: HybridState) -> dict:
        messages = deepcopy(state.get("messages"))
        original_query = messages[
            -3
        ].content  # always take query that was used for the most recent run
        response = synthesize(original_query=original_query, all_generation=messages)
        return {"messages": [json.dumps(response.get("final_synthesis"))]}

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
        graph = graph_builder.compile()
        return graph

    async def invoke_graph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    async def _handle(self, query: str) -> str:
        result = await self.invoke_graph(query)
        result = result.get("messages")[-1].content
        return result

    async def _handle_stream(self, query: str):
        searches = {
            "rag": self.rag_stream_search,
            "grag": self.grag_stream_search,
            "agent": self.agent_stream_search,
        }
        queue: asyncio.Queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(self._stream_component(source, search, query, queue))
            for source, search in searches.items()
        ]
        combined_outputs = {source: "" for source in searches}
        remaining = len(tasks)

        while remaining:
            source, kind, content = await queue.get()
            if kind == "chunk":
                combined_outputs[source] += content
                yield self._stream_event(source, kind, content)
            elif kind == "final":
                combined_outputs[source] = content
                yield self._stream_event(source, kind, content)
            elif kind == "error":
                yield self._stream_event(source, kind, content)
            elif kind == "done":
                remaining -= 1
                yield self._stream_event(source, kind)

        await asyncio.gather(*tasks)

        messages = [
            HumanMessage(content=query),
            HumanMessage(content=combined_outputs["rag"]),
            HumanMessage(content=combined_outputs["grag"]),
            HumanMessage(content=combined_outputs["agent"]),
        ]
        final = synthesize(original_query=query, all_generation=messages)
        final = json.dumps(final.get("final_synthesis"))
        yield self._stream_event("hybrid", "final", final)

    def handle(self, query: str) -> Any:
        if self.stream:
            return self._handle_stream(query)
        return self._handle(query)
