"""
Module with RAG system for AI search in GeneNetwork
Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

import dspy
from gnais.search.classification import classify_search
from gnais.search.corpus import create_ensemble_retriever
from gnais.search.tools import MemoryTools
from typing import Any


class RAG(dspy.Signature):
    input_text: str = dspy.InputField(desc="Query and instructions")
    chat_history: list = dspy.InputField(desc="History of conversation")
    context: list = dspy.InputField(desc="Background information")
    feedback: str = dspy.OutputField(
        desc="System response to the query that has a list of detailed answers and the final answer"
    )



_SYSTEM_PROMPT = """You excel at addressing search query using the context and chat history you have. You do not make mistakes.
Extract answers to the query below from the context and chat history. Use the chat history before moving to the context.
Provide links associated with each RDF entity. To build links you must replace RDF prefixes by namespaces.
Here is the mapping of prefixes and namespaces:
gn => http://rdf.genenetwork.org/v1/id
gnc => http://rdf.genenetwork.org/v1/category
owl => http://www.w3.org/2002/07/owl
gnt => http://rdf.genenetwork.org/v1/term
skos = http://www.w3.org/2004/02/skos/core
xkos => http://rdf-vocabulary.ddialliance.org/xkos
rdf => http://www.w3.org/1999/02/22-rdf-syntax-ns
rdfs => http://www.w3.org/2000/01/rdf-schema
taxon => http://purl.uniprot.org/taxonomy
dcat => http://www.w3.org/ns/dcat
dct => http://purl.org/dc/terms
xsd => http://www.w3.org/2001/XMLSchema
sdmx-measure => http://purl.org/linked-data/sdmx/2009/measure
qb => http://purl.org/linked-data/cube
pubmed => http://rdf.ncbi.nlm.nih.gov/pubmed
v => http://www.w3.org/2006/vcard/ns
foaf => http://xmlns.com/foaf/0.1
geoSeries => http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc

All links pointing to specific traits should be translated to CD links using the trait id and the dataset name.
Original trait link: https://rdf.genenetwork.org/v1/id/trait_16339
Trait id: 16339
Dataset name: BXDPublish
New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish

Format your entire response as valid HTML. Use tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>. Do not wrap the response in markdown code blocks.
"""


async def rag_search(
    query: str,
    docs: Any,
    chroma_db: Any,
    memory: Any = None,
    user_id: str = "default_user",
):
    if classify_search(query).get("decision") == "keyword":
        ensemble_retriever = create_ensemble_retriever(
            chroma_db=chroma_db, docs=docs, keyword_weight=0.7
        )
    else:
        ensemble_retriever = create_ensemble_retriever(
            chroma_db=chroma_db, docs=docs
        )

    chat_history, memory_tools = [], None
    if memory is not None:
        memory_tools = MemoryTools(memory)
        try:
            memories = memory_tools.search_memories(query, user_id=user_id)
            if memories and not any(
                marker in memories for marker in ("No relevant memories found", "Error searching memories")
            ):
                chat_history = [memories]
        except Exception:
            pass

    prompt = f"{_SYSTEM_PROMPT}\nQuery: {query}"

    predict = dspy.streamify(
        dspy.Predict(RAG),
        stream_listeners=[
            dspy.streaming.StreamListener(
                signature_field_name="feedback", allow_reuse=True
            )
        ],
        include_final_prediction_in_output_stream=True,
    )

    full_response = ""
    async for value in predict(
            input_text=prompt,
            chat_history=chat_history,
            context=ensemble_retriever.invoke(query),
    ):
        if isinstance(value, dspy.Prediction):
            yield {"final": value.feedback}
        else:
            chunk = getattr(value, "chunk", str(value))
            full_response += chunk
            yield chunk

    # Persist the interaction in memory after streaming completes
    if memory_tools is not None:
        try:
            memory_tools.store_memory(
                f"User query: {query}\nSystem response: {full_response}",
                user_id=user_id,
            )
        except Exception:
            pass
