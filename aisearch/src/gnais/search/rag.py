"""Module with RAG system for AI search in GeneNetwork"""

import asyncio
from typing import Any

import dspy
from gnais.search.tools import with_memory
from gnais.search.prompts import GENERAL_SYSTEM_PROMPT
from typing import Any


class RAG(dspy.Signature):
    input_text: str = dspy.InputField(desc="Query and instructions")
    chat_history: list = dspy.InputField(desc="History of conversation")
    context: list = dspy.InputField(desc="Background information")
    feedback: str = dspy.OutputField(
        desc="""System response to the query — answer ONLY from the context and chat history provided.
HTML answer. Link rules:
- ONLY use <a href> for full web URLs that literally appear in the context.
- NEVER invent RDF/IRI links.
- For entities (datasets, genes, traits, strains) use PLAIN TEXT or <strong> tags.
- If a URL is not in the context, do NOT create a link for it.
- Always fully expand prefixes (.e.g. gn:87e8288b-697c-5b77-a944-bc27d89b19c3 -> https://rdf.genenetwork.org/v1/id/87e8288b-697c-5b77-a944-bc27d89b19c3)

Fully expand prefixes according to the following map:

PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX ex: <http://example.org/stuff/1.0/>
PREFIX fabio: <http://purl.org/spar/fabio/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX generif: <http://www.ncbi.nlm.nih.gov/gene?cmd=Retrieve&dopt=Graphics&list_uids=>
PREFIX genotype: <http://rdf.genenetwork.org/v1/genotype/>
PREFIX gn: <http://rdf.genenetwork.org/v1/id/>
PREFIX gnc: <http://rdf.genenetwork.org/v1/category/>
PREFIX gnt: <http://rdf.genenetwork.org/v1/term/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX phenotype: <http://rdf.genenetwork.org/v1/phenotype/>
PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
PREFIX publication: <http://rdf.genenetwork.org/v1/publication/>
PREFIX pubmed: <http://rdf.ncbi.nlm.nih.gov/pubmed/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX taxon: <https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=>
PREFIX up: <http://purl.uniprot.org/core/>
PREFIX xkos: <http://rdf-vocabulary.ddialliance.org/xkos#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
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
    system_prompt: str = GENERAL_SYSTEM_PROMPT,
    user_id: str = "default_user",
    memory=None,
    chat_history: list = [],
):
    prompt = f"{system_prompt}\nQuery: {query}"
    yield {"status": "Fetching context…"}
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
