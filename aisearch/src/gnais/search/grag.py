"""Module with GraphRAG system for AI search in GeneNetwork"""

import asyncio
import functools
import dspy
from gnais.config import Config
from gnais.search import tools
from gnais.search.tools import with_memory, build_schema_hint, sparql_fetch
from gnais.search.prompts import GENERAL_SYSTEM_PROMPT, SPARQL_SYSTEM_PROMPT


class KeywordSPARQLGenerator(dspy.Signature):
    """
Extract the essential keywords from the user's query, then generate fast,
efficient SPARQL SELECT queries that avoid Virtuoso timeouts (504 errors).
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
- To extract information about a specific trait, find the closest matches in SNAPSHOT_OBJECTS and use them to infer the proper RDF subject for the trait.
- To get trait page or URL, use the predicate `gnt:has_trait_page` with the trait as subject. Its object correspond to the URL. Never build trait URLs manually.
- To get publication for a trait, search for object having the predicate `dcterms:references` and the trait as subject.
- To get highest LOD score and corresponding marker and additive effect, use the predicates `gnt:lod_score`, `gnt:locus`, and `gnt:additive`, respectively.
- Mean trait measurements can be accessed by looking up object with predicate gnt:mean and the trait as subject.
- Trait descriptions can be accessed via the predicate `dcterms:description`
- When you have the description of a trait, you should use regex pattern in the object term with the predicate `dcterms:description` to find the corresponding trait (subject)
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
    keywords: str = dspy.OutputField(desc="Comma-separated essential keywords from the query")
    sparql_queries: list[str] = dspy.OutputField(
        desc="Top 10 valid SPARQL SELECT queries with PREFIX declarations."
    )


class GraphRAG(dspy.Signature):
    original_query: str = dspy.InputField(desc="Query provided")
    sparql_results: str = dspy.InputField(desc="JSON results from the SPARQL query")
    chat_history: list = dspy.InputField(desc="History of conversation")
    feedback: str = dspy.OutputField(
        desc="System response to the query with detailed answers and the final answer, formatted as valid HTML using tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>.  Links from sparql results can only be formed by valid IRIs and NOT literals.  Use the orinial query, sparql results and chat history when answering."
    )


_KW_SPARQL_GEN = dspy.Predict(KeywordSPARQLGenerator)

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
    grag_prompt = f"{system_prompt}\nQuery: {query}"
    sparql_prompt = f"{SPARQL_SYSTEM_PROMPT}\nQuery: {query}"
    schema_hint = build_schema_hint(sparql_url)
    yield {"status": "Generating keywords and SPARQL queries…"}
    loop = asyncio.get_running_loop()
    combined = await loop.run_in_executor(
        tools.LLM_EXECUTOR,
        functools.partial(
            _KW_SPARQL_GEN,
            original_query=sparql_prompt,
            schema_hint=schema_hint,
        ),
    )
    keywords = getattr(combined, "keywords", "")
    sparql_queries = getattr(combined, "sparql_queries", [])
    yield {"status": f"Extracted keywords: {keywords}"}
    if sparql_queries is None:
        sparql_queries = []
    for i, query in enumerate(sparql_queries):
        yield {"status": f"({i+1}). sparql query: {query}"}

    yield {"status": "Querying knowledge graph…"}
    sparql_results = await sparql_fetch(sparql_queries, sparql_url)

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
