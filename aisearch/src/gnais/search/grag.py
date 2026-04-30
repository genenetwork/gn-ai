"""Module with GraphRAG system for AI search in GeneNetwork"""

import sys
import dspy
from gnais.search.classification import extract_keywords
from gnais.search.tools import with_memory, _ONTOLOGY_HINTS
from gnais.search.prompts import GRAG_SYSTEM_PROMPT
from SPARQLWrapper import JSON, SPARQLWrapper


def _run_sparql_queries(sparql_url: str, sparql_queries: list[str]) -> str:
    sparql = SPARQLWrapper(sparql_url)
    sparql.setReturnFormat(JSON)
    results = []
    for i, sparql_query in enumerate(sparql_queries, 1):
        try:
            sparql.setQuery(sparql_query)
            result = sparql.queryAndConvert()
            bindings = result.get("results", {}).get("bindings", [])
            results.append(f"Query {i} succeeded ({len(bindings)} rows): {bindings}")
        except Exception as e:
            results.append(
                f"Query {i} failed: {e}\nQuery was:\n{sparql_query}"
            )
    return "\n\n".join(results)


class SPARQLGenerator(dspy.Signature):
    """Generate valid SPARQL SELECT queries from a natural language question.

    CRITICAL RULES:
    1. EVERY query MUST start with the PREFIX declarations provided above.
    2. Only use prefixes that are declared above. Do NOT invent new prefixes.
    3. Use the classes and properties from the schema to build the query.
    4. Prefer simple SELECT ?s ?p ?o patterns when exploring.
    5. When looking for traits, use gnt:has_trait_page to get direct URLs.
    """

    original_query: str = dspy.InputField(desc="User query")
    classes_info: str = dspy.InputField(desc="Available classes")
    properties_info: str = dspy.InputField(desc="Available properties")
    sparql_queries: list[str] = dspy.OutputField(
        desc="As many and exhaustive SPARQL SELECT queries that you can generate and that can retrieve all relevant information necessary to provide detailed answer to the user query."
    )


class GraphRAG(dspy.Signature):
    original_query: str = dspy.InputField(desc="Query provided")
    sparql_results: str = dspy.InputField(desc="JSON results from the SPARQL query")
    chat_history: list = dspy.InputField(desc="History of conversation")
    feedback: str = dspy.OutputField(
        desc="System response to the query that has a list of detailed answers and the final answer"
    )


_SPARQL_GEN = dspy.Predict(SPARQLGenerator)

_GRAG_STREAM = dspy.streamify(
    dspy.Predict(GraphRAG),
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="feedback")
    ],
    include_final_prediction_in_output_stream=True,
)


@with_memory(memory_type="grag")
async def graph_rag_search(
    query: str,
    sparql_url: str,
    system_prompt: str = GRAG_SYSTEM_PROMPT,
    memory=None,
    user_id: str = "default_user",
    chat_history: list = [],
):
    keywords_pred = extract_keywords(query)
    keywords = getattr(keywords_pred, "keywords", str(keywords_pred))

    prompt = f"{system_prompt}\n{keywords}"

    sparql_gen = _SPARQL_GEN(
        original_query=prompt,
        classes_info=_ONTOLOGY_HINTS,
        properties_info="See ontology hints above.",
    )
    sparql_queries = getattr(sparql_gen, "sparql_queries", [])
    if sparql_queries is None:
        sparql_queries = []

    sparql_results = _run_sparql_queries(sparql_url, sparql_queries)

    async for value in _GRAG_STREAM(
        original_query=prompt,
        sparql_results=sparql_results,
        chat_history=chat_history,
    ):
        if isinstance(value, dspy.Prediction):
            yield {"final": value.feedback}
        else:
            yield value.chunk
