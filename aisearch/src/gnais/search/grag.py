"""Module with GraphRAG system for AI search in GeneNetwork"""

import sys
import dspy
from gnais.search.classification import extract_keywords
from gnais.search.tools import with_memory
from gnais.utils import fetch_schema
from SPARQLWrapper import JSON, SPARQLWrapper

_SYSTEM_PROMPT = """\
You excel at addressing search query using the context and chat history you have. You do not make mistakes.
Extract answers to the query below from the context and chat history. Use the chat history before moving to the context.
Provide links associated with each RDF entity. To build links you must replace RDF prefixes by namespaces using the schema provided.
All links pointing to specific traits should be translated to CD links using the trait id and the dataset name.
Original trait link: https://rdf.genenetwork.org/v1/id/trait_16339
Trait id: 16339
Dataset name: BXDPublish
New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish

Format your entire response as valid HTML. Use tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>. Do not wrap the response in markdown code blocks."""


def _run_sparql_queries(sparql_url: str, sparql_queries: list[str]) -> str:
    try:
        sparql = SPARQLWrapper(sparql_url)
        sparql.setReturnFormat(JSON)
        results = []
        for sparql_query in sparql_queries:
            sparql.setQuery(sparql_query)
            results.append(str(sparql.queryAndConvert()["results"]["bindings"]))
        return str(results)
    except Exception as e:
        return f"Query failed: {str(e)}"


class SPARQLGenerator(dspy.Signature):
    """Generate a SPARQL SELECT query from a natural language question.
    Use the provided schema to construct valid queries."""

    original_query: str = dspy.InputField(desc="User query")
    classes_info: str = dspy.InputField(desc="Mapping for available classes")
    properties_info: str = dspy.InputField(desc="Mapping for available properties")
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
    rdf_classes, rdf_properties = fetch_schema(sparql_url)

    keywords_pred = extract_keywords(query)
    keywords = getattr(keywords_pred, "keywords", str(keywords_pred))

    prompt = f"{_SYSTEM_PROMPT}\n{keywords}"

    sparql_gen = dspy.Predict(SPARQLGenerator)(
        original_query=prompt,
        classes_info=rdf_classes,
        properties_info=rdf_properties,
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
