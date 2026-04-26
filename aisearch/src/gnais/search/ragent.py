"""Hybrid system of RAG and Agent for GeneNetwork's AI Search"""

__all__ = (
    "hybrid_search",
    "Synthesis",
    "SearchResult",
    "StreamEvent",
)

import asyncio
from functools import partial
from typing import Any
from typing_extensions import TypedDict

import dspy
from gnais.search.agent import agent_search
from gnais.config import Config
from gnais.search.grag import graph_rag_search
from gnais.search.rag import rag_search
from gnais.search.classification import classify_search
from gnais.search.corpus import get_docs, get_chroma_db, create_ensemble_retriever


class SearchResult(TypedDict):
    """Standardized search result schema with flexible results array"""

    query: str
    status: str  # "success" or "partial_success"
    summary: str
    results: list[dict]  # Each result has: type, name, description, url, extra
    note: str


class StreamEvent(TypedDict):
    """SSE-ready event emitted by the hybrid search stream."""
    source: str
    kind: str
    content: str


class Synthesis(dspy.Signature):
    """Synthesize the final response from all search components.

    Format your entire response as valid HTML. Use tags such as
    <p>, <ul>, <li>, <a>, <strong>, <em>, <h3>, and <br>.
    Do not wrap the response in markdown code blocks.

    Structure the HTML as follows:
    - A short status banner (<div class="status-success"> or "status-partial">).
    - A summary paragraph (<p class="summary">).
    - For each group of results, an <h3> heading with the type
      (e.g. "Genes", "Traits", "Datasets") followed by a <ul class="card-list">.
      Each item should be a <li class="card-item"> containing:
        - <div class="card-title"><a href="URL">Name</a></div>
        - <div class="card-description">Description</div> (optional)
        - <div class="card-meta">Extra info</div> (optional)
    - If no results were found, a <div class="note-box"> explaining why.
    - Any additional notes in a <div class="note-box"> at the end.
    - If there's need for a table, add a table with (<table class="data">)
    """

    original_query: str = dspy.InputField()
    all_generation: list[str] = dspy.InputField()
    feedback: str = dspy.OutputField(
        desc="Final synthesized response formatted as valid HTML"
    )


_synthesize = dspy.streamify(
    dspy.Predict(Synthesis),
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="feedback", allow_reuse=True)
    ],
    include_final_prediction_in_output_stream=True,
)

_RAG_CHROMA_DB = get_chroma_db(
    chroma_db_path=Config.DB_PATH,
    embed_model="Qwen/Qwen3-Embedding-0.6B",
)
_RAG_DOCS = get_docs(Config.CORPUS_PATH)
_RETRIEVER_KW = create_ensemble_retriever(
    chroma_db=_RAG_CHROMA_DB, docs=_RAG_DOCS, keyword_weight=0.7
)
_RETRIEVER_SEM = create_ensemble_retriever(
    chroma_db=_RAG_CHROMA_DB, docs=_RAG_DOCS, keyword_weight=0.5
)


async def _rag_search(query: str, user_id: str = "default_user"):
    retriever = (
        _RETRIEVER_KW
        if classify_search(query).get("decision") == "keyword"
        else _RETRIEVER_SEM
    )
    async for item in rag_search(query, retriever=retriever, user_id=user_id):
        yield item


_grag_search = partial(graph_rag_search, sparql_url=Config.SPARQL_ENDPOINT)
_agent_search = partial(agent_search, sparql_url=Config.SPARQL_ENDPOINT)


async def _stream_component(
    source: str, search_func: Any, query: str, queue: asyncio.Queue, user_id: str
) -> None:
    try:
        async for chunk in search_func(query, user_id=user_id):
            if isinstance(chunk, dict) and "final" in chunk:
                await queue.put(
                    StreamEvent(source=source, kind="final", content=chunk["final"])
                )
            else:
                await queue.put(
                    StreamEvent(source=source, kind="chunk", content=str(chunk))
                )
    except Exception as exc:
        await queue.put(StreamEvent(source=source, kind="error", content=str(exc)))
    finally:
        await queue.put(StreamEvent(source=source, kind="done", content=""))


async def hybrid_search(query: str, user_id: str = "default_user"):
    """Run hybrid search with concurrent RAG, GraphRAG, and Agent.

    Yields :class:`StreamEvent` dicts for progress from each component,
    followed by a final synthesis event with ``source="hybrid"``.
    """
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_stream_component("rag", _rag_search, query, queue, user_id)),
        asyncio.create_task(_stream_component("grag", _grag_search, query, queue, user_id)),
        asyncio.create_task(_stream_component("agent", _agent_search, query, queue, user_id)),
    ]

    combined_outputs = {"rag": "", "grag": "", "agent": ""}
    remaining = len(tasks)

    while remaining:
        event = await queue.get()
        yield event

        if event["kind"] == "final":
            combined_outputs[event["source"]] = event["content"]
        elif event["kind"] == "done":
            remaining -= 1

    await asyncio.gather(*tasks)

    messages = [
        query,
        combined_outputs["rag"],
        combined_outputs["grag"],
        combined_outputs["agent"],
    ]

    synthesis_text = ""
    has_chunks = False
    async for value in _synthesize(original_query=query, all_generation=messages):
        if isinstance(value, dspy.Prediction):
            synthesis_text = value.feedback
        else:
            chunk = getattr(value, "chunk", str(value))
            synthesis_text += chunk
            has_chunks = True
            yield StreamEvent(source="synthesis", kind="chunk", content=chunk)

    if not has_chunks and synthesis_text:
        yield StreamEvent(source="synthesis", kind="chunk", content=synthesis_text)

    yield StreamEvent(source="hybrid", kind="final", content=synthesis_text)
