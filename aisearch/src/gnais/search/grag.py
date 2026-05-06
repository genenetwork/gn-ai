"""Module with GraphRAG system for AI search in GeneNetwork"""

import asyncio
import dspy
from gnais.search.classification import extract_keywords
from gnais.config import Config
from gnais.search.tools import with_memory, build_schema_hint, sparql_fetch
from gnais.search.prompts import GENERAL_SYSTEM_PROMPT, SPARQL_SYSTEM_PROMPT
from SPARQLWrapper import JSON, SPARQLWrapper


class SPARQLGenerator(dspy.Signature):
    """
Generate fast, efficient SPARQL SELECT queries that avoid Virtuoso timeouts (504 errors).
Use the schema hint to match keywords to exact properties and classes.

CRITICAL PERFORMANCE RULES (to prevent 504s):
1. Always add `LIMIT` - start with `LIMIT 50`, increase only if needed. Never omit `LIMIT`.
2. Never use `SELECT *` - list only the variables you actually need.
3. Avoid expensive operations: no Cartesian products, no cross joins, no full graph scans.
4. Use specific FILTER patterns that leverage indexes:
   - Prefer `STRSTARTS(?label, "prefix")` over `CONTAINS` or regex.
   - Avoid `FILTER regex(...)` - it disables indexes.
   - Use `FILTER(?value = "exact")` or `IN` with small lists.
5. Prefer property paths over multiple joins when traversing a chain.
6. Use VALUES blocks for small sets of constants instead of UNION or OPTIONAL.
7. Avoid ORDER BY on large result sets - if needed, combine with `LIMIT` and a narrow `WHERE` clause.
8. Never use nested subqueries unless absolutely necessary; flatten them.
9. Use `OPTIONAL` only for truly optional patterns – otherwise, use a simple triple pattern.
10. Limit the number of generated queries - output at most 10 (not 20) per request.

SPARQL SYNTAX RULES (remember):
- Literal properties (e.g., gnt:gene_symbol, dct:title) hold strings/numbers. Use `FILTER(?literal = "value")`, not `?o a <Class>`.
- Object properties (e.g., gnt:has_phenotype_trait) link to resources. Chain with `?s gnt:has_phenotype_trait ?o . ?o rdf:type <Class>`.
- Only use properties listed in the provided schema. Do NOT invent new ones.
- EVERY query MUST start with the PREFIX declarations from the schema.

Example of an efficient query:

```
PREFIX gnt: <http://rdf.genenetwork.org/v1/term/>
PREFIX gnc: <http://rdf.genenetwork.org/v1/category/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?gene ?symbol WHERE {
    ?gene rdf:type gnc:gene .
    ?gene gnt:gene_symbol ?symbol .
    FILTER(STRSTARTS(?symbol, "Shh"))
} LIMIT 50
```
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
        desc="System response to the query with detailed answers and the final answer, formatted as valid HTML using tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>.  Links from sparql results can only be formed by valid IRIs and NOT literals.  Use the sparql results and chat history when answering."
    )


_SPARQL_GEN = dspy.Predict(SPARQLGenerator)

_GRAG_STREAM = dspy.streamify(
    dspy.Predict(GraphRAG),
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="feedback")
    ],
    include_final_prediction_in_output_stream=True,
)


# Warm schema cache at import time to avoid blocking the first request
_ = build_schema_hint(Config.SPARQL_ENDPOINT)


@with_memory(memory_type="grag")
async def graph_rag_search(
    query: str,
    sparql_url: str,
    system_prompt: str = GENERAL_SYSTEM_PROMPT,
    memory=None,
    user_id: str = "default_user",
    chat_history: list = [],
):
    yield {"status": "Extracting keywords…"}
    keywords_pred = await asyncio.to_thread(extract_keywords, query)
    keywords = getattr(keywords_pred, "keywords", str(keywords_pred))

    grag_prompt = f"{system_prompt}\nQuery: {query}"
    sparql_prompt = f"{SPARQL_SYSTEM_PROMPT}\nQuery: {query}\nEssential keywords in query: {keywords}"
    yield {"status": f"Extracted essential keywords: {keywords}"}
    schema_hint = build_schema_hint(sparql_url)
    yield {"status": f"Generating sparql queries…"}
    sparql_gen = await asyncio.to_thread(
        _SPARQL_GEN,
        original_query=sparql_prompt,
        schema_hint=schema_hint,
    )
    sparql_queries = getattr(sparql_gen, "sparql_queries", [])
    if sparql_queries is None:
        sparql_queries = []
    for i, query in enumerate(sparql_queries):
        yield {"status": f"({i+1}). sparql query: {query}"}

    yield {"status": "Querying knowledge graph…"}
    sparql_results = await asyncio.to_thread(
        sparql_fetch, sparql_queries, sparql_url
    )

    yield {"status": "Streaming response…"}
    async for value in _GRAG_STREAM(
        original_query=grag_prompt,
        sparql_results=sparql_results,
        chat_history=chat_history,
    ):
        if isinstance(value, dspy.Prediction):
            yield {"final": value.feedback}
        else:
            yield value.chunk
