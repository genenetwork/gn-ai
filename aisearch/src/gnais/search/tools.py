import asyncio
import concurrent.futures
import functools
import logging
import os
import random
from typing import Any

import dspy
import httpx
import redis

_redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Dedicated thread pool for LLM inference so heavy model calls don't
# saturate the default asyncio executor used for lighter I/O work.
LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() // 2, thread_name_prefix="llm-worker"
)

# mem0's internal history store can spew sqlite transaction warnings;
# suppress them so they don't clutter CLI output.
for _mem0_logger in ("mem0", "mem0.memory", "mem0.memory.main"):
    logging.getLogger(_mem0_logger).addFilter(
        lambda record: "Failed to add history" not in record.getMessage()
    )


async def _exec_sparql(
    sparql_uri: str,
    query: str,
    max_retries: int = 3,
    base_delay: float = 1,
) -> dict:
    """Execute a single SPARQL query with retry + exponential jitter via httpx."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    for attempt in range(max_retries):
        try:
            resp = await client.post(
                sparql_uri,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (504, 503, 502) and attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2**attempt) + random.uniform(0, 1))
                continue
            raise
    return {}


@functools.lru_cache(maxsize=64)
def _fetch_schema(sparql_uri: str) -> tuple[set[str], set[str], set[str]]:
    """Fetch literal and object properties from the live Virtuoso endpoint.

    Returns (literal_props, object_props, snapshot_objs).
    The three queries run concurrently in a thread pool.
    """

    def _bindings(result: dict, var: str) -> set[str]:
        return {
            b[var]["value"]
            for b in result.get("results", {}).get("bindings", [])
            if b.get(var)
        }

    queries = {
        "literal": """
SELECT DISTINCT ?p
  FROM <http://rdf.genenetwork.org/v1>
WHERE { ?s ?p ?o . FILTER isLiteral(?o) }
        """,
        "object": """
SELECT DISTINCT ?p
  FROM <http://rdf.genenetwork.org/v1>
WHERE { ?s ?p ?o . FILTER isIRI(?o) }
        """,
        "snapshot": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT SAMPLE(?obj) AS ?o
  FROM <http://rdf.genenetwork.org/v1>
WHERE {
    ?subject skos:member ?obj .
    BIND(skos:member AS ?predicate)
    BIND(LCASE(REPLACE(STR(?obj), "^([^_]*_[^_]*_).*$", "$1")) AS ?stem)
    FILTER (?subject != ?obj)
    FILTER (0.1 > <SHORT_OR_LONG::bif:rnd> (10, ?subject, ?predicate))
}
GROUP BY ?subject ?predicate ?stem
        """,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = {
            name: executor.submit(
                lambda uri, query: asyncio.run(_exec_sparql(uri, query)),
                sparql_uri,
                query,
            )
            for name, query in queries.items()
        }
        results = {name: fut.result() for name, fut in futures.items()}

    literal_props = _bindings(results["literal"], "p")
    object_props = _bindings(results["object"], "p")
    snapshot_objs = _bindings(results["snapshot"], "o")

    # Cap sizes so the prompt doesn't bloat to thousands of tokens
    literal_props = set(list(literal_props)[:100])
    object_props = set(list(object_props)[:100])
    snapshot_objs = set(list(snapshot_objs)[:50])

    return literal_props, object_props, snapshot_objs


# Compact namespace → prefix map for prompt output
_PREFIX_MAP = {
    "http://rdf.genenetwork.org/v1/term/": "gnt",
    "http://rdf.genenetwork.org/v1/category/": "gnc",
    "http://rdf.genenetwork.org/v1/id/": "gn",
    "http://purl.org/dc/terms/": "dct",
    "http://www.w3.org/ns/dcat#": "dcat",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/2004/02/skos/core#": "skos",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://purl.org/linked-data/cube#": "qb",
    "http://purl.org/linked-data/sdmx/2009/measure#": "sdmx-measure",
    "http://rdf-vocabulary.ddialliance.org/xkos#": "xkos",
    "https://schema.org/": "schema",
    "http://rdf.ncbi.nlm.nih.gov/pubmed/": "pubmed",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "http://purl.org/spar/fabio/": "fabio",
    "http://prismstandard.org/namespaces/basic/2.0/": "prism",
}


def _uri_to_qname(uri: str) -> str:
    """Convert a full URI to a prefixed name, or return the URI in angle brackets."""
    for ns, prefix in sorted(_PREFIX_MAP.items(), key=lambda x: -len(x[0])):
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):]}"
    return f"<{uri}>"


def build_schema_hint(sparql_uri: str) -> str:
    """Build a compact schema hint from the live Virtuoso endpoint.

    Cached in Redis for 1 week so multiple workers / restarts share it.
    """
    cache_key = f"gn:schema_hint:{sparql_uri}"
    cached = _redis.get(cache_key)
    if cached:
        return cached

    literal_props, object_props, snapshot_objs = _fetch_schema(sparql_uri)
    hint = f"""=== GENENETWORK SCHEMA (from Virtuoso) ===
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX gn: <http://rdf.genenetwork.org/v1/id/>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX gnc: <http://rdf.genenetwork.org/v1/category/>
PREFIX gnt: <http://rdf.genenetwork.org/v1/term/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>


LITERAL PROPERTIES (object is a string/number/date):
{" ,".join([_uri_to_qname(uri) for uri in literal_props])}


OBJECT PROPERTIES (object is a URI / another resource):
{" ,".join([_uri_to_qname(uri) for uri in object_props])}


SNAPSHOT_OBJECTS (use to build targeted queries):
{" ,".join([_uri_to_qname(uri) for uri in snapshot_objs])}


CRITICAL RULES:
1. Only use properties listed above. Do NOT invent new ones.
2. Literal properties give strings/numbers — use FILTER, not ?o a ...
3. Object properties link to other resources — you can chain ?o a <Class>.
4. Do NOT use taxon: for species. Use gn:Mus_musculus, gn:Rattus_norvegicus, gn:Homo_sapiens, etc.
    """
    _redis.setex(cache_key, 604800, hint)  # 1 week
    return hint


class QueryTranslation(dspy.Signature):
    """Compare object snapshot in schema hint to keywords in the original query to find best semantic matches.
    Use matches to generate valid SPARQL SELECT queries that can retrieve relevant information for the query.

    CRITICAL SCHEMA RULES (derived from GeneNetwork RDF transforms):
    1. Every query MUST start with the PREFIX declarations. Only use declared prefixes.
    2. ALWAYS include `FROM <http://rdf.genenetwork.org/v1>` between SELECT and WHERE.
    3. Strain URIs: use gn:set_BXD, gn:set_B6D2F1, gn:set_HMDP — NEVER gn:BXD, gn:B6D2F1, gn:HMDP.
    4. gnc:phenotype is metadata only (abbreviation, description, lab_code, submitter, contributor). Phenotype TRAITS (with mean, locus, lod_score, additive, sequence, has_trait_page) are gnc:phenotype_trait.
    5. Probesets and DNA markers use gnt:chr for chromosome. Genes use gnt:chromosome.
    6. gnt:has_uniprot_id, gnt:has_homologene_id, gnt:has_kegg_id, gnt:has_omim_id, gnt:has_chebi_id, gnt:has_pub_chem_id exist on gnc:probeset, NOT on gnc:gene.
    7. gnt:has_align_id, gnt:has_protein_id, gnt:has_rgd_id, gnc:has_kg_id, gnc:has_unigen_id exist on gnc:gene, NOT on gnc:dna_marker. Markers use skos:prefLabel or skos:altLabel for names.
    8. gnt:locus on phenotype_trait contains chromosome positions (e.g. "1-59904011"), NOT phenotype names. Never FILTER gnt:locus for trait names like "liver_weight". Instead link trait -> has_phenotype -> phenotype and filter gnt:abbreviation.
    9. gnt:symbol is for probesets. gnt:gene_symbol is for genes and strains.
    10. Datasets are dcat:Dataset. They have dct:title, gnt:has_strain, gnt:has_tissue_info, gnt:has_samples, gnt:has_summary, gnt:has_citation, gnt:has_contributors, gnt:has_case_info, gnt:has_platform_info, gnt:has_specifics, gnt:has_data_processing_info, gnt:has_experiment_design, gnt:has_experiment_type, gnt:has_genotype_files.
    11. Do NOT use taxon: for species. Use gn:Mus_musculus, gn:Rattus_norvegicus, gn:Homo_sapiens, etc.
    12. Gene chips / platforms are skos:Concept with skos:inScheme gnc:gene_chip. They have gnt:has_go_tree_value, gnt:has_geo_series_id.
    13. Mapping methods and averaging methods are skos:Concept in gnc:mapping_method and gnc:avg_method schemes.
    14. gnt:has_trait_page gives the URL directly. Never build trait URLs manually.
    15. Leverage as many schema hints as possible.
    16. To extract information about a specific trait, find the closest matches in SNAPSHOT_OBJECTS and use them to infer the proper RDF subject for the trait.
    17. To get trait page or URL, use the predicate `gnt:has_trait_page` with the trait as subject. Its object correspond to the URL. Never build trait URLs manually.
    18. To get publication for a trait, search for object having the predicate `dcterms:references` and the trait as subject.
    19. To get highest LOD score and corresponding marker and additive effect, use the predicates `gnt:lod_score`, `gnt:locus`, and `gnt:additive`, respectively.
    20. Mean trait measurements can be accessed by looking up object with predicate gnt:mean and the trait as subject.
    21. Trait descriptions can be accessed via the predicate `dcterms:description`
    22. When you have the description of a trait, you should use regex pattern in the object term with the predicate `dcterms:description` to find the corresponding trait (subject).

    When querying SPARQL, prefer fast, efficient SPARQL SELECT queries
    that avoid Virtuoso timeouts (504 errors).

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
    """

    original_query: str = dspy.InputField(desc="User query")
    schema_hint: str = dspy.InputField(desc="GeneNetwork schema from Virtuoso")
    translated_queries: list[str] = dspy.OutputField(
        desc="Top 7 valid most relevant SPARQL queries with the highest likelihood of getting non-empty results."
    )


async def sparql_fetch(
    sparql_queries: list[str],
    sparql_uri: str,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> str:
    """Execute *sparql_queries* concurrently against *sparql_uri*."""
    if not sparql_queries:
        return "No SPARQL queries to run."

    async def _fetch_one(query: str, idx: int) -> str:
        try:
            result = await _exec_sparql(sparql_uri, query, max_retries, base_delay)
            bindings = result.get("results", {}).get("bindings", [])
            return f"Query {idx} succeeded ({len(bindings)} rows): {bindings}"
        except Exception as e:
            return f"Query {idx} failed: {e}\nQuery was:\n{query}"

    tasks = [_fetch_one(q, i) for i, q in enumerate(sparql_queries)]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)


@functools.lru_cache(maxsize=64)
def make_sparql_fetch_tool(sparql_uri: str) -> dspy.Tool:
    def _fetch(query: str) -> Any:
        schema_hint = build_schema_hint(sparql_uri)
        pred = dspy.Predict(QueryTranslation)(
            original_query=query,
            schema_hint=schema_hint,
        )
        sparql_queries = pred.get("translated_queries") if pred else []
        if not sparql_queries:
            return "No SPARQL queries generated."
        future = LLM_EXECUTOR.submit(
            asyncio.run, sparql_fetch(sparql_queries, sparql_uri)
        )
        return future.result()

    return dspy.Tool(
        name="fetch_data",
        desc="Fetch RDF data around GeneNetwork data through SPARQL",
        args={
            "query": {
                "type": "string",
                "desc": "SPARQL query to run to fetch relevant data",
            },
        },
        func=_fetch,
    )


def _check_link(url: str) -> str:
    """Check whether a URL is reachable.

    Returns a short status string suitable for feeding back to the LLM.
    """
    try:
        response = httpx.Client(follow_redirects=True, timeout=10).head(url)
        ok = response.is_success
    except Exception:
        ok = False

    return f"{'Valid URL' if ok else 'Invalid URL'} URL: {url}"


check_link = dspy.Tool(
    name="check_link",
    desc="Check whether a URL is valid and reachable. Call this BEFORE putting any URL in an <a href>. Returns 'Valid URL: ...' or 'Invalid URL: ...'.",
    args={
        "url": {
            "type": "string",
            "desc": "URL or link to check",
        },
    },
    func=_check_link,
)


# KLUDGE: For now this is lifted from:
# <https://dspy.ai/tutorials/mem0_react_agent/>
class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory):
        self.memory = memory

    def store_memory(
        self, content: str, user_id: str, run_id: str, metadata: dict = {}
    ) -> str:
        """Store information in memory."""
        try:
            self.memory.add(
                content, user_id=user_id, run_id=run_id, metadata=metadata, infer=True
            )
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def search_memories(
        self, query: str, user_id: str, run_id: str, limit: int = 10
    ) -> str:
        """Search for relevant memories."""
        results = self.memory.search(
            query, filters={"user_id": user_id, "run_id": run_id}, top_k=limit
        )
        chat_history = ""
        if results and (memories := results.get("results")):
            for memory in memories:
                if query := memory.get("metadata", {}).get("query"):
                    chat_history += f"Query: {query}\n"
                chat_history += (
                    f"\nMemory({memory['updated_at']}): {memory['memory']}\n"
                )
        return chat_history.strip()

    def get_all_memories(self, user_id: str, run_id: str, filters: dict = {}) -> str:
        """Get all memories for a user."""
        try:
            results = self.memory.get_all(
                filters={"user_id": user_id, "run_id": run_id}
            )
            if not results or not results.get("results"):
                return "No memories found for this user."

            memory_text = "All memories for user:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Update an existing memory."""
        try:
            self.memory.update(memory_id, new_content)
            return f"Updated memory with new content: {new_content}"
        except Exception as e:
            return f"Error updating memory: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            return "Memory deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {str(e)}"


def with_memory(memory_type: str = "interaction"):
    """Decorator factory that injects chat_history from mem0 and persists the interaction after streaming."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            query = kwargs.get("query")
            memory = kwargs.get("memory")
            user_id = kwargs.get("user_id")

            # Pre: search memories and build chat_history
            chat_history = []
            memory_tools = None
            if memory is not None:
                memory_tools = MemoryTools(memory)
                memories = memory_tools.search_memories(
                    query, user_id=user_id, run_id=memory_type
                )
                if memories:
                    chat_history = [memories]

            kwargs["chat_history"] = chat_history
            async for value in func(*args, **kwargs):
                if isinstance(value, dict):
                    feedback = str(value.get("final"))
                    if memory_tools and feedback:
                        # Fire-and-forget: don't block the stream on SQLite writes
                        asyncio.create_task(
                            asyncio.to_thread(
                                memory_tools.store_memory,
                                feedback,
                                user_id=user_id,
                                run_id=memory_type,
                                metadata={"query": query},
                            )
                        )
                yield value

        return wrapper

    return decorator


class Route(dspy.Signature):
    """
    Choose the most efficient model to handle the task assigned to the program between the LLMs available.
    The most efficient model is the LLM that can follow the instructions defined in the program signature with good fidelity while saving on unneccessary cost.
    """

    signature: str = dspy.InputField(desc="The signature of the DSPy program")
    models: list[str] = dspy.InputField(desc="The 2 models or LLMs available")
    best_model: str = dspy.OutputField(desc="The most efficient model for the task")


def route_model(options: list[dspy.LM]):
    """Decorator function that helps choose the right model for a DSPy module"""

    def decorator(func: dspy.Module):
        @functools.wraps(func)
        def wrapper(**kwargs):
            signature = func.signature
            models = [model.__dict__["model"] for model in options]
            best_model = dspy.Predict(Route)(signature=signature, models=models).get(
                "best_model"
            )
            func.set_lm(best_model)
            return func(**kwargs)

        return wrapper

    return decorator
