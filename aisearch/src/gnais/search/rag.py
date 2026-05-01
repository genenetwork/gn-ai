"""
Module with RAG system for AI search in GeneNetwork
Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

import asyncio

import dspy
from gnais.search.tools import with_memory
from gnais.search.prompts import RAG_SYSTEM_PROMPT
from typing import Any


class RAG(dspy.Signature):
    input_text: str = dspy.InputField(desc="Query and instructions")
    chat_history: list = dspy.InputField(desc="History of conversation")
    context: list = dspy.InputField(desc="Background information")
    feedback: str = dspy.OutputField(
        desc="""System response to the query — answer ONLY from the context provided.
HTML answer. Link rules:
- ONLY use <a href> for full web URLs that literally appear in the context.
- NEVER invent RDF/IRI links (e.g., gn:BXD, http://rdf.genenetwork.org/v1/id/BXD).
- For entities (datasets, genes, traits, strains) use PLAIN TEXT or <strong> tags.
- If a URL is not in the context, do NOT create a link for it."""
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

@with_memory(memory_type="rag")
async def rag_search(
    query: str,
    retriever: Any,
    system_prompt: str = RAG_SYSTEM_PROMPT,
    user_id: str = "default_user",
    memory: Any = None,
    chat_history: list = [],
):
    prompt = f"{system_prompt}\nQuery: {query}"

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
