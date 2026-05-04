import asyncio
import functools
import logging
from typing import Any

import dspy
import requests
from SPARQLWrapper import JSON, SPARQLWrapper

# mem0's internal history store can spew sqlite transaction warnings;
# suppress them so they don't clutter CLI output.
for _mem0_logger in ("mem0", "mem0.memory", "mem0.memory.main"):
    logging.getLogger(_mem0_logger).addFilter(
        lambda record: "Failed to add history" not in record.getMessage()
    )


# ---------------------------------------------------------------------------
# Ground-truth schema from Virtuoso (memoized)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _fetch_schema(sparql_uri: str) -> tuple[set[str], set[str]]:
    """Fetch literal and object properties from the live Virtuoso endpoint.

    Returns (literal_props, object_props) where each is a set of full URIs.
    """
    sparql = SPARQLWrapper(sparql_uri)
    sparql.setReturnFormat(JSON)

    literal_query = """
        SELECT DISTINCT ?p
        FROM <http://rdf.genenetwork.org/v1>
        WHERE { ?s ?p ?o . FILTER isLiteral(?o) }
    """
    sparql.setQuery(literal_query)
    lit_result = sparql.queryAndConvert()
    literal_props = {
        b["p"]["value"]
        for b in lit_result.get("results", {}).get("bindings", [])
        if b.get("p")
    }

    object_query = """
        SELECT DISTINCT ?p
        FROM <http://rdf.genenetwork.org/v1>
        WHERE { ?s ?p ?o . FILTER isIRI(?o) }
    """
    sparql.setQuery(object_query)
    obj_result = sparql.queryAndConvert()
    object_props = {
        b["p"]["value"]
        for b in obj_result.get("results", {}).get("bindings", [])
        if b.get("p")
    }

    snapshot_query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT SAMPLE(?obj) AS ?o
        WHERE {
        ?subject skos:member ?obj .
        BIND(skos:member AS ?predicate)
        BIND(LCASE(REPLACE(STR(?obj), "^([^_]*_[^_]*_).*$", "$1")) AS ?stem)
        FILTER (?subject != ?obj)
        FILTER (0.1 > <SHORT_OR_LONG::bif:rnd> (10, ?subject, ?predicate))
        }
        GROUP BY ?subject ?predicate ?stem
    """
    sparql.setQuery(snapshot_query)
    snapshot_result = sparql.queryAndConvert()
    snapshot_objs = {
        b["o"]["value"]
        for b in snapshot_result.get("results", {}).get("bindings", [])
        if b.get("o")
    }

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
    """Build a compact schema hint from the live Virtuoso endpoint."""
    literal_props, object_props, snapshot_objs = _fetch_schema(sparql_uri)
    return f"""=== GENENETWORK SCHEMA (from Virtuoso) ===
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
    5. gnt:has_trait_page gives the URL directly. Never build trait URLs manually.
    """

# ---------------------------------------------------------------------------
# mem0 / memory tools (unchanged)
# ---------------------------------------------------------------------------

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
                    query,
                    user_id=user_id,
                    run_id=memory_type
                )
                if memories:
                    chat_history = [memories]

            kwargs["chat_history"] = chat_history
            async for value in func(*args, **kwargs):
                if isinstance(value, dict):
                    feedback = str(value.get("final"))
                    if memory_tools and feedback:
                        memory_tools.store_memory(
                            f"Query: {query} \nFeedback: {feedback}",
                            user_id=user_id,
                            run_id=memory_type
                        )
                yield value
        return wrapper
    return decorator


# KLUDGE: For now this is lifted from:
# <https://dspy.ai/tutorials/mem0_react_agent/>
class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str, run_id: str, metadata: dict = {}) -> str:
        """Store information in memory."""
        try:
            self.memory.add(content, user_id=user_id, run_id=run_id, metadata=metadata)
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def search_memories(self, query: str, user_id: str, run_id: str, limit: int = 20) -> str:
        """Search for relevant memories."""
        results = self.memory.search(query, user_id=user_id, run_id=run_id, limit=limit)
        if results and results.get("results"):
            return "\n".join([r["memory"] for r in results["results"]])
        return ""

    def get_all_memories(self, user_id: str, run_id: str, filters: dict = {}) -> str:
        """Get all memories for a user."""
        try:
            results = self.memory.get_all(user_id=user_id, run_id=run_id, filters=filters)
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


def _check_link(url: str) -> str:
    """Check whether a URL is reachable.

    Returns a short status string suitable for feeding back to the LLM.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        ok = response.ok
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


class QueryTranslation(dspy.Signature):
    """Translate natural language query to SPARQL SELECT following closely instructions below.
    Compare object snapshot in schema hint to keywords in the original query to find best semantic matches.
    Use matches to generate valid SPARQL SELECT queries that can retrieve relevant information for the query.
    CRITICAL:
    1. Every query MUST start with the PREFIX declarations. Only use declared prefixes.
    2. Leverage as many schema hints as possible.
    """

    original_query: str = dspy.InputField(desc="User query")
    schema_hint: str = dspy.InputField(desc="GeneNetwork schema from Virtuoso")
    translated_queries: list[str] = dspy.OutputField(
        desc="Top 10 valid SPARQL SELECT query with PREFIX declarations."
    )


_translate_query = dspy.Predict(QueryTranslation)


def sparql_fetch(query: str, sparql_uri: str) -> Any:
    schema_hint = build_schema_hint(sparql_uri)
    sparql_queries = _translate_query(
        original_query=query,
        schema_hint=schema_hint,
    ).get("translated_queries")

    sparql = SPARQLWrapper(sparql_uri)
    sparql.setReturnFormat(JSON)
    results = []
    for i, sparql_query in enumerate(sparql_queries):
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


def make_sparql_fetch_tool(sparql_uri: str) -> dspy.Tool:
    def _fetch(query: str) -> Any:
        return sparql_fetch(query, sparql_uri)

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
