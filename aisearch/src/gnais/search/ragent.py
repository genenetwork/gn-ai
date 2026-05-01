"""Hybrid system of RAG and Agent for GeneNetwork's AI Search"""

__all__ = (
    "hybrid_search",
    "Synthesis",
    "SearchResult",
    "StreamEvent",
)

import asyncio
from functools import lru_cache, partial
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
    - If there's need for a table, add a table with a <table class="data">
    - If no results were found, a <div class="note-box"> explaining why.
    - Any additional notes in a <div class="note-box"> at the end.
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

@lru_cache(maxsize=8)
def _get_retrievers():
    """Build (and cache) the RAG ensemble retrievers.  This is expensive
    because it tokenizes the entire corpus for BM25, so we only do it
    when hybrid_search is actually invoked."""
    chroma_db = get_chroma_db(
        chroma_db_path=Config.DB_PATH,
        embed_model="Qwen/Qwen3-Embedding-0.6B",
    )
    docs = get_docs(Config.CORPUS_PATH)
    return {
        "kw": create_ensemble_retriever(
            chroma_db=chroma_db, docs=docs, keyword_weight=0.7
        ),
        "sem": create_ensemble_retriever(
            chroma_db=chroma_db, docs=docs, keyword_weight=0.5
        ),
    }


async def _rag_search(query: str, user_id: str = "default_user", memory=None):
    retrievers = _get_retrievers()
    retriever = (
        retrievers["kw"]
        if classify_search(query).get("decision") == "keyword"
        else retrievers["sem"]
    )
    async for item in rag_search(query=query, retriever=retriever, user_id=user_id, memory=memory):
        yield item


async def _grag_search(query: str, user_id: str = "default_user", memory=None):
    async for chunk in graph_rag_search(
        query=query, sparql_url=Config.SPARQL_ENDPOINT, memory=memory, user_id=user_id
    ):
        yield chunk


_agent_search = partial(agent_search, sparql_url=Config.SPARQL_ENDPOINT)


async def _stream_component(
    source: str, search_func: Any, queue: asyncio.Queue, **kwargs
) -> None:
    try:
        async for chunk in search_func(**kwargs):
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


async def hybrid_search(query: str, user_id: str = "default_user", memory=None):
    """Run hybrid search with concurrent RAG, GraphRAG, and Agent.

    Yields :class:`StreamEvent` dicts for progress from each component,
    followed by a final synthesis event with ``source="hybrid"``.
    """
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_stream_component("rag", _rag_search, queue, query=query, user_id=user_id, memory=memory)),
        asyncio.create_task(_stream_component("grag", _grag_search, queue, query=query, user_id=user_id, memory=memory)),
        asyncio.create_task(_stream_component("agent", _agent_search, queue, query=query, user_id=user_id, memory=memory)),
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
