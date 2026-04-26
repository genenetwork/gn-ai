"""Module with GraphRAG system for AI search in GeneNetwork"""

import sys
import dspy
from gnais.search.classification import extract_keywords
from gnais.search.tools import with_memory, _QUERY_HINTS, _SPARQL_PREFIXES
from SPARQLWrapper import JSON, SPARQLWrapper

_SYSTEM_PROMPT = """Answer from SPARQL results. Work with partial data; do not apologize for query errors.
Links: expand ALL turtle prefixes before using in <a href>.
Examples (not complete): pubmed:→http://rdf.ncbi.nlm.nih.gov/pubmed/ taxon:→http://purl.uniprot.org/taxonomy/
gn:→http://rdf.genenetwork.org/v1/id gnc:→http://rdf.genenetwork.org/v1/category gnt:→http://rdf.genenetwork.org/v1/term dcat:→http://www.w3.org/ns/dcat dct:→http://purl.org/dc/terms rdfs:→http://www.w3.org/2000/01/rdf-schema skos:→http://www.w3.org/2004/02/skos/core
Trait links: use the URL from gnt:has_trait_page. Never build trait URLs manually.
Format as HTML using <p>,<ul>,<li>,<a>,<strong>,<em>,<br>. No markdown blocks.
"""


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



async def graph_rag_search(
    query: str,
    sparql_url: str,
    memory=None,
    user_id: str = "default_user",
    chat_history: list = [],
):
    keywords_pred = extract_keywords(query)
    keywords = getattr(keywords_pred, "keywords", str(keywords_pred))

    prompt = f"{_SYSTEM_PROMPT}\n{keywords}"

    sparql_gen = dspy.Predict(SPARQLGenerator)(
        original_query=prompt,
        classes_info=_SPARQL_PREFIXES + "\n" + _QUERY_HINTS,
        properties_info="See ontology hints above.",
    )
    sparql_queries = getattr(sparql_gen, "sparql_queries", [])
    if sparql_queries is None:
        sparql_queries = []

    sparql_results = _run_sparql_queries(sparql_url, sparql_queries)

    predict = dspy.streamify(
        dspy.Predict(GraphRAG),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="feedback")
        ],
        include_final_prediction_in_output_stream=True,
    )

    async for value in predict(
        original_query=prompt,
        sparql_results=sparql_results,
        chat_history=chat_history,
    ):
        if isinstance(value, dspy.Prediction):
            yield {"final": value.feedback}
        else:
            yield value.chunk
