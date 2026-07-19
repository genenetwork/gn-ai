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
from gnais.config import Config

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


def build_schema_hint(sparql_uri: str) -> str:
    """Build a compact schema hint from the live Virtuoso endpoint.

    Cached in Redis for 1 week so multiple workers / restarts share it.
    """
    cache_key = f"gn:schema_hint:{sparql_uri}"
    cached = _redis.get(cache_key)
    if cached:
        return cached

    hint = """
=== HIGH LEVEL STRUCTURE OF THE GENENETWORK RDF GRAPH ===
```mermaid
graph TD
    A[gnc:resource_classification_scheme<br/>a skos:ConceptScheme] -->|xkos:levels| B[gnc:taxonomic_family<br/>a xkos:ClassificationLevel<br/>xkos:depth 1]
    A -->|xkos:levels| C[gnc:species<br/>a xkos:ClassificationLevel<br/>xkos:depth 2]
    A -->|xkos:levels| D[gnc:population_category<br/>a xkos:ClassificationLevel<br/>xkos:depth 3]
    A -->|xkos:levels| E[gnc:set<br/>a xkos:ClassificationLevel<br/>xkos:depth 4]

    B -->|xkos:nextLevel| C
    C -->|xkos:nextLevel| D
    D -->|xkos:nextLevel| E

    B -->|gnt:has_taxonomic_family| F[gni:family_*<br/>no explicit rdf:type]
    C -->|skos:member| G[gni:species_*<br/>a gnc:species]
    D -->|gnt:has_reference_population| H[gni:population_*<br/>a gnc:reference_population]
    E -->|skos:member| I[gni:set_*<br/>a gnc:set]

    F -->|gnt:has_species| G
    G -->|gnt:has_strain| I
    H -->|gnt:has_strain| I

    I -->|gnt:has_species| G
    I -->|gnt:has_reference_population| H

    G -->|gnt:has_uniprot_taxon_id| J[uniprot taxon ID]

    I -->|gnt:has_genotype_data| K[gni:dataset_*]
    I -->|gnt:has_phenotype_data| K
    I -->|gnt:has_probeset_data| K
    I -->|gnt:uses_mapping_method| L[gni:mapping_method_*]

    K -->|gnt:has_molecular_trait| M[gni:trait_*]
    K -->|gnt:has_phenotype_trait| M
    K -->|gnt:uses_genechip| N[gni:platform_*]
    K -->|gnt:uses_normalization_method| O[gni:avg_method_*]
    K -->|gnt:has_strain| I
```

=== DETAILED SCHEMA OF GENENETWORK RDF DATABASE ===
```turtle
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xkos: <http://rdf-vocabulary.ddialliance.org/xkos#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix gnc: <http://rdf.genenetwork.org/v1/category/> .
@prefix gni: <http://rdf.genenetwork.org/v1/id/> .
@prefix gnt: <http://rdf.genenetwork.org/v1/term/> .
@prefix uniprot: <http://purl.uniprot.org/taxonomy/> .

# -------------------------------------------------
# Schema and its four levels
# -------------------------------------------------

gnc:resource_classification_scheme a skos:ConceptScheme ;
    rdfs:label "GeneNetwork Resource Classification Scheme" ;
    skos:prefLabel "GeneNetwork Resource Classification Scheme" ;
    skos:definition "A hierarchical classification scheme for organizing GeneNetwork resources by dataset type, resource set (inbredset group), or species." ;
    xkos:numberOfLevels 4 ;
    xkos:levels gnc:taxonomic_family , gnc:species , gnc:population_category , gnc:set .

# Level containers. Each is an xkos:ClassificationLevel and also a class for its members.
gnc:taxonomic_family a xkos:ClassificationLevel , rdfs:Class ;
    rdfs:label "Family" ;
    skos:prefLabel "Family" ;
    skos:definition "An organizational classification level used in GeneNetwork to group resources into families." ;
    xkos:depth 1 ;
    xkos:nextLevel gnc:species ;
    skos:inScheme gnc:resource_classification_scheme .

gnc:species a xkos:ClassificationLevel , rdfs:Class ;
    rdfs:label "Species" ;
    skos:prefLabel "Species" ;
    skos:definition "A classification level that associates a given resource to a species in GeneNetwork." ;
    xkos:depth 2 ;
    xkos:previousLevel gnc:taxonomic_family ;
    xkos:nextLevel gnc:population_category ;
    skos:inScheme gnc:resource_classification_scheme .

gnc:population_category a xkos:ClassificationLevel , rdfs:Class ;
    rdfs:label "Population Category" ;
    skos:prefLabel "Population Category" ;
    xkos:depth 3 ;
    xkos:previousLevel gnc:species ;
    xkos:nextLevel gnc:set ;
    skos:inScheme gnc:resource_classification_scheme .

gnc:set a xkos:ClassificationLevel , rdfs:Class ;
    rdfs:label "Set" ;
    skos:prefLabel "Set" ;
    skos:definition "A category representing groups of genetically related strains or individuals (inbred sets, recombinant inbred lines, etc.)." ;
    xkos:depth 4 ;
    xkos:previousLevel gnc:population_category ;
    skos:inScheme gnc:resource_classification_scheme .

# -------------------------------------------------
# Classes for non-level member instances
# -------------------------------------------------

gnc:reference_population a rdfs:Class ;
    rdfs:label "Reference Population" ;
    rdfs:comment "Class of the entities linked from gnc:population_category via gnt:has_reference_population." .

# Note: gni:family_* entities are linked from gnc:taxonomic_family but do not carry an explicit rdf:type in the source data.

# -------------------------------------------------
# Properties
# -------------------------------------------------

# XKOS / SKOS structural properties
xkos:levels a rdf:Property ;
    rdfs:label "levels" ;
    rdfs:domain skos:ConceptScheme ;
    rdfs:range xkos:ClassificationLevel .

xkos:numberOfLevels a rdf:Property ;
    rdfs:label "number of levels" ;
    rdfs:domain skos:ConceptScheme ;
    rdfs:range xsd:integer .

xkos:nextLevel a rdf:Property ;
    rdfs:label "next level" ;
    rdfs:domain xkos:ClassificationLevel ;
    rdfs:range xkos:ClassificationLevel .

xkos:previousLevel a rdf:Property ;
    rdfs:label "previous level" ;
    rdfs:domain xkos:ClassificationLevel ;
    rdfs:range xkos:ClassificationLevel .

xkos:depth a rdf:Property ;
    rdfs:label "depth" ;
    rdfs:domain xkos:ClassificationLevel ;
    rdfs:range xsd:integer .

skos:member a rdf:Property ;
    rdfs:label "member" ;
    rdfs:domain skos:Collection ;
    rdfs:range skos:Concept ;
    rdfs:comment "Used here by gnc:species and gnc:set to link to their member instances." .

# GeneNetwork-specific membership properties

gnt:has_taxonomic_family a rdf:Property ;
    rdfs:label "has taxonomic family" ;
    rdfs:domain gnc:taxonomic_family ;
    rdfs:range rdfs:Resource ;
    rdfs:comment "Links the taxonomic_family level to gni:family_* entities." .

gnt:has_reference_population a rdf:Property ;
    rdfs:label "has reference population" ;
    rdfs:domain [ a rdfs:Class ; owl:unionOf ( gnc:population_category gnc:set ) ] ;
    rdfs:range gnc:reference_population ;
    rdfs:comment "Links the population_category level to gni:population_* entities; also used from gni:set_* back to populations." .

gnt:has_species a rdf:Property ;
    rdfs:label "has species" ;
    rdfs:domain [ a rdfs:Class ; owl:unionOf ( gnc:taxonomic_family gnc:set ) ] ;
    rdfs:range gnc:species ;
    rdfs:comment "Links family entities to species, and set entities back to species." .

gnt:has_strain a rdf:Property ;
    rdfs:label "has strain" ;
    rdfs:domain [ a rdfs:Class ; owl:unionOf ( gnc:species gnc:reference_population ) ] ;
    rdfs:range gnc:set ;
    rdfs:comment "Links species or reference populations to gni:set_* entities." .

# GeneNetwork-specific descriptive properties on instances

gnt:has_uniprot_taxon_id a rdf:Property ;
    rdfs:label "has UniProt taxon ID" ;
    rdfs:domain gnc:species ;
    rdfs:range uniprot:Taxon .

gnt:short_name a rdf:Property ;
    rdfs:label "short name" ;
    rdfs:domain gnc:species ;
    rdfs:range xsd:string .

gnt:has_family_order_id a rdf:Property ;
    rdfs:label "has family order ID" ;
    rdfs:domain rdfs:Resource ;
    rdfs:range xsd:integer ;
    rdfs:comment "Used on gni:family_* entities for ordering." .

gnt:has_population_order_id a rdf:Property ;
    rdfs:label "has population order ID" ;
    rdfs:domain gnc:reference_population ;
    rdfs:range xsd:integer .

gnt:has_set_code a rdf:Property ;
    rdfs:label "has set code" ;
    rdfs:domain gnc:set ;
    rdfs:range xsd:string .

# Dataset linkage properties from gni:set_* entities

gnt:has_genotype_data a rdf:Property ;
    rdfs:label "has genotype data" ;
    rdfs:domain gnc:set ;
    rdfs:range dcat:Dataset .

gnt:has_phenotype_data a rdf:Property ;
    rdfs:label "has phenotype data" ;
    rdfs:domain gnc:set ;
    rdfs:range dcat:Dataset .

gnt:has_probeset_data a rdf:Property ;
    rdfs:label "has probeset data" ;
    rdfs:domain gnc:set ;
    rdfs:range dcat:Dataset .

gnt:uses_mapping_method a rdf:Property ;
    rdfs:label "uses mapping method" ;
    rdfs:domain gnc:set ;
    rdfs:range skos:Concept ;
    rdfs:comment "Refers to concepts in the gnc:mapping_method SKOS scheme." .

# Dataset-level properties

gnt:has_molecular_trait a rdf:Property ;
    rdfs:label "has molecular trait" ;
    rdfs:domain dcat:Dataset ;
    rdfs:range skos:Concept ;
    rdfs:comment "Refers to concepts in the gnc:molecular_trait collection." .

gnt:has_phenotype_trait a rdf:Property ;
    rdfs:label "has phenotype trait" ;
    rdfs:domain dcat:Dataset ;
    rdfs:range skos:Concept ;
    rdfs:comment "Used by phenotype datasets." .

gnt:uses_genechip a rdf:Property ;
    rdfs:label "uses genechip" ;
    rdfs:domain dcat:Dataset ;
    rdfs:range rdfs:Resource ;
    rdfs:comment "Refers to gni:platform_* entities." .

gnt:uses_normalization_method a rdf:Property ;
    rdfs:label "uses normalization method" ;
    rdfs:domain dcat:Dataset ;
    rdfs:range skos:Concept ;
    rdfs:comment "Refers to concepts in the gnc:avg_method SKOS scheme." .

# Common annotation properties

rdfs:label a rdf:Property ;
    rdfs:label "label" ;
    rdfs:domain rdfs:Resource ;
    rdfs:range xsd:string .

skos:prefLabel a rdf:Property ;
    rdfs:label "preferred label" ;
    rdfs:domain rdfs:Resource ;
    rdfs:range xsd:string .

skos:altLabel a rdf:Property ;
    rdfs:label "alternative label" ;
    rdfs:domain rdfs:Resource ;
    rdfs:range xsd:string .

skos:definition a rdf:Property ;
    rdfs:label "definition" ;
    rdfs:domain rdfs:Resource ;
    rdfs:range xsd:string .

dcterms:description a rdf:Property ;
    rdfs:label "description" ;
    rdfs:domain rdfs:Resource ;
    rdfs:range rdf:HTML ;
    rdfs:comment "On gni:set_* entities this is often an rdf:HTML literal." .

dcterms:created a rdf:Property ;
    rdfs:label "created date" ;
    rdfs:domain dcat:Dataset ;
    rdfs:range xsd:datetime .

dcterms:identifier a rdf:Property ;
    rdfs:label "identifier" ;
    rdfs:domain dcat:Dataset ;
    rdfs:range xsd:string .

dcterms:title a rdf:Property ;
    rdfs:label "title" ;
    rdfs:domain dcat:Dataset ;
    rdfs:range xsd:string .
```

=== EXAMPLES OF SUCCESSFUL QUERIES WITH KEY QUERY PATTERNS ===

Question: List all classification levels in order
SPARQL Query:
```sparql
SELECT ?level ?depth ?label
WHERE {
  gnc:resource_classification_scheme xkos:levels ?level .
  ?level xkos:depth ?depth ;
         skos:prefLabel ?label .
}
ORDER BY ?depth
```

Question: Find all sets belonging to a species
SPARQL Query:
```sparql
SELECT ?species ?set
WHERE {
  ?species a gnc:species ;
           gnt:has_strain ?set .
}
```

Question: Find all datasets reachable from a species (through sets)
SPARQL Query:
```sparql
SELECT ?species ?set ?dataset
WHERE {
  ?species a gnc:species ;
           gnt:has_strain ?set .
  ?set gnt:has_genotype_data|gnt:has_phenotype_data|gnt:has_probeset_data ?dataset .
}
```

Question: Find all sets for a given reference population
SPARQL Query:
```sparql
SELECT ?population ?set
WHERE {
  ?population a gnc:reference_population ;
              gnt:has_strain ?set .
}
```

Question: Traverse family → species → set → dataset
SPARQL Query:
```sparql
SELECT ?family ?species ?set ?dataset
WHERE {
  gnc:taxonomic_family gnt:has_taxonomic_family ?family .
  ?family gnt:has_species ?species .
  ?species gnt:has_strain ?set .
  ?set gnt:has_genotype_data|gnt:has_phenotype_data|gnt:has_probeset_data ?dataset .
}
```
    """
    _redis.setex(cache_key, 604800, hint)  # 1 week
    return hint


class QueryTranslation(dspy.Signature):
    """Use schema hints to generate valid SPARQL SELECT queries that can retrieve relevant information for the query.

    CRITICAL SCHEMA RULES (derived from GeneNetwork RDF schema):
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
def make_sparql_fetch_tool(
    sparql_uri: str, lm: dspy.LM = Config.DEFAULT_LLM
) -> dspy.Tool:
    def _fetch(query: str) -> Any:
        schema_hint = build_schema_hint(sparql_uri)
        pred = dspy.Predict(QueryTranslation)
        pred.set_lm(lm)
        pred = pred(
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
    The most efficient model is the LLM that can follow the instructions defined in the program with good fidelity while saving on unneccessary cost.
    """

    task: str = dspy.InputField(desc="The task of the DSPy program")
    models: list[str] = dspy.InputField(desc="The 2 models or LLMs available")
    best_model: str = dspy.OutputField(desc="The most efficient model for the task")


class RoutedModule(dspy.Module):
    def __init__(self, module: dspy.Module, options: dict[str, dspy.LM]):
        super().__init__()
        self.module = module
        self.options = options
        self.router = dspy.Predict(Route)

    def choose_model(self) -> str:
        task = str(self.module.signature)
        models = list(self.options.keys())
        return self.router(task=task, models=models).get("best_model")

    def forward(self, **kwargs):
        chosen_model = self.choose_model()
        print(
            f"Choice made: {chosen_model} for {self.module.__dict__['signature'].__name__}"
        )
        self.module.set_lm(self.options[chosen_model])
        return self.module(**kwargs)

    def get(self, field_name, default=None):
        return getattr(self.module, field_name, default)


def route_model(
    options: dict[str, dspy.LM] = {
        Config.DEFAULT_MODEL: Config.DEFAULT_LLM,
        Config.ALTERNATIVE_MODEL: Config.ALTERNATIVE_LLM,
    }
):
    """Decorator that returns a dspy.Module wrapping the routed module."""

    def decorator(module: dspy.Module):
        return RoutedModule(module, options)

    return decorator
