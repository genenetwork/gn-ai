"""Module with GraphRAG system for AI search in GeneNetwork"""

import dspy
import time
from gnais.search.classification import extract_keywords
from gnais.search.tools import with_memory, build_schema_hint
from gnais.search.prompts import GENERAL_SYSTEM_PROMPT, SPARQL_SYSTEM_PROMPT
from SPARQLWrapper import JSON, SPARQLWrapper


def _run_sparql_queries(sparql_url: str, sparql_queries: list[str]) -> str:
    sparql = SPARQLWrapper(sparql_url)
    sparql.setReturnFormat(JSON)
    results = []
    for i, sparql_query in enumerate(sparql_queries):
        try:
            sparql.setQuery(sparql_query)
            result = sparql.queryAndConvert()
            bindings = result.get("results", {}).get("bindings", [])
            results.append(f"Query {i} succeeded ({len(bindings)} rows): {bindings}")
            # NOTE: Break communication with endpoint between queries
            # to prevent connection closure to endpoint
            time.sleep(5)
        except Exception as e:
            results.append(
                f"Query {i} failed: {e}\nQuery was:\n{sparql_query}"
            )
    return "\n\n".join(results)


class SPARQLGenerator(dspy.Signature):
    """Generate valid SPARQL SELECT queries from a natural language query following closely instructions below.
    Compare object snapshot in schema hint to keywords in the original query to find best semantic matches.
    Use matches to generate valid SPARQL SELECT queries that can retrieve relevant information for the query.
    CRITICAL SPARQL RULES:
    1. Literal properties (e.g., gnt:gene_symbol, dct:title) hold strings/numbers. Use FILTER, not ?o a ...
    2. Object properties (e.g., gnt:has_phenotype_trait) link to other resources. You can chain ?o a <Class>.
    3. gnt:has_trait_page gives the direct URL; never construct trait URLs manually.
    4. Only use properties listed in the provided schema. Do NOT invent new ones.
    5. EVERY query MUST start with the PREFIX declarations.
    6. ALWAYS try to build FAST and EFFICIENT sparql queries.
    """

    original_query: str = dspy.InputField(desc="User query")
    schema_hint: str = dspy.InputField(desc="GeneNetwork schema from Virtuoso")
    sparql_queries: list[str] = dspy.OutputField(
        desc="Top 20 valid SPARQL SELECT queries to retrieve relevant information and provide detailed answers to original query using schema hints."
    )


class GraphRAG(dspy.Signature):
    original_query: str = dspy.InputField(desc="Query provided")
    sparql_results: str = dspy.InputField(desc="JSON results from the SPARQL query")
    chat_history: list = dspy.InputField(desc="History of conversation")
    feedback: str = dspy.OutputField(
        desc="System response to the query with detailed answers and the final answer, formatted as valid HTML using tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>.  Links from sparql results can only be formed by valid IRIs and NOT literals."
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
    system_prompt: str = GENERAL_SYSTEM_PROMPT,
    memory=None,
    user_id: str = "default_user",
    chat_history: list = [],
):
    keywords_pred = extract_keywords(query)
    keywords = getattr(keywords_pred, "keywords", str(keywords_pred))

    grag_prompt = f"{system_prompt}\nQuery: {query}"
    sparql_prompt = f"{SPARQL_SYSTEM_PROMPT}\nQuery: {query}\nEssential keywords in query: {keywords}"
    schema_hint = build_schema_hint(sparql_url)
    sparql_gen = _SPARQL_GEN(
        original_query=sparql_prompt,
        schema_hint=schema_hint,
    )
    sparql_queries = getattr(sparql_gen, "sparql_queries", [])
    if sparql_queries is None:
        sparql_queries = []

    sparql_results = _run_sparql_queries(sparql_url, sparql_queries)

    async for value in _GRAG_STREAM(
        original_query=grag_prompt,
        sparql_results=sparql_results,
        chat_history=chat_history,
    ):
        if isinstance(value, dspy.Prediction):
            yield {"final": value.feedback}
        else:
            yield value.chunk
