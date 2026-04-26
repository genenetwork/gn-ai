"""
Module with RAG system for AI search in GeneNetwork
Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

import asyncio

import dspy
from gnais.search.tools import with_memory
from typing import Any


class RAG(dspy.Signature):
    input_text: str = dspy.InputField(desc="Query and instructions")
    chat_history: list = dspy.InputField(desc="History of conversation")
    context: list = dspy.InputField(desc="Background information")
    feedback: str = dspy.OutputField(
        desc="System response to the query that has a list of detailed answers and the final answer"
    )


_RAG_STREAM = dspy.streamify(
    dspy.Predict(RAG),
    stream_listeners=[
        dspy.streaming.StreamListener(
            signature_field_name="feedback", allow_reuse=True
        )
    ],
    include_final_prediction_in_output_stream=True,
)

_SYSTEM_PROMPT = """Answer from the context and chat history. Use chat history first.
Links: expand ALL turtle prefixes before using in <a href>.
Examples (not complete): pubmed:→http://rdf.ncbi.nlm.nih.gov/pubmed/ taxon:→http://purl.uniprot.org/taxonomy/
gn:→http://rdf.genenetwork.org/v1/id gnc:→http://rdf.genenetwork.org/v1/category gnt:→http://rdf.genenetwork.org/v1/term dcat:→http://www.w3.org/ns/dcat dct:→http://purl.org/dc/terms rdfs:→http://www.w3.org/2000/01/rdf-schema skos:→http://www.w3.org/2004/02/skos/core
Trait links: use the URL from gnt:has_trait_page. Never build trait URLs manually.
Format as HTML using <p>,<ul>,<li>,<a>,<strong>,<em>,<br>. No markdown blocks.
"""


@with_memory
async def rag_search(
    query: str,
    retriever: Any,
    memory: Any = None,
    user_id: str = "default_user",
    chat_history: list = [],
):
    prompt = f"{_SYSTEM_PROMPT}\nQuery: {query}"

    context = await asyncio.to_thread(retriever.invoke, query)

    async for value in _RAG_STREAM(
            input_text=prompt,
            chat_history=chat_history,
            context=context,
    ):
        if isinstance(value, dspy.Prediction):
            yield {"final": value.feedback}
        else:
            yield getattr(value, "chunk", str(value))
