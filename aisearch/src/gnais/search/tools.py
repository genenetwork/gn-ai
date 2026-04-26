from mem0 import Memory
import datetime
import functools
import logging
import os
from typing import Any

import dspy
from SPARQLWrapper import JSON, SPARQLWrapper

# mem0's internal history store can spew sqlite transaction warnings;
# suppress them so they don't clutter CLI output.
for _mem0_logger in ("mem0", "mem0.memory", "mem0.memory.main"):
    logging.getLogger(_mem0_logger).addFilter(
        lambda record: "Failed to add history" not in record.getMessage()
    )


def with_memory(func):
    """Decorator that injects chat_history from mem0 and persists the interaction after streaming."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        query = kwargs.get("query", "")
        memory = kwargs.get("memory")
        user_id = kwargs.get("user_id", "default_user")

        # Pre: search memories and build chat_history
        chat_history = []
        memory_tools = None
        if memory is not None:
            memory_tools = MemoryTools(memory)
            try:
                memories = memory_tools.search_memories(query, user_id=user_id)
                if memories and not any(
                    marker in memories
                    for marker in ("No relevant memories found", "Error searching memories")
                ):
                    chat_history = [memories]
            except Exception:
                pass

        kwargs["chat_history"] = chat_history

        # Run the original async generator and collect the full response
        full_response = ""
        async for chunk in func(*args, **kwargs):
            if isinstance(chunk, dict) and "final" in chunk:
                full_response = chunk["final"]
            yield chunk
        if memory_tools is not None and full_response:
            try:
                memory_tools.store_memory(
                    f"User query: {query}\nSystem response: {full_response}",
                    user_id=user_id,
                )
            except Exception:
                pass

    return wrapper


class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """Store information in memory."""
        try:
            self.memory.add(content, user_id=user_id)
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """Search for relevant memories."""
        try:
            results = self.memory.search(query, filters={"user_id": user_id}, limit=limit)
            if not results:
                return "No relevant memories found."

            memory_text = "Relevant memories found:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """Get all memories for a user."""
        try:
            results = self.memory.get_all(filters={"user_id": user_id})
            if not results:
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


# Prefix table for SPARQL and link expansion.
# When you see pubmed:123 use http://rdf.ncbi.nlm.nih.gov/pubmed/123
# When you see taxon:10090 use http://purl.uniprot.org/taxonomy/10090
_SPARQL_PREFIXES = """PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX gn: <http://rdf.genenetwork.org/v1/id/>
PREFIX gnc: <http://rdf.genenetwork.org/v1/category/>
PREFIX gnt: <http://rdf.genenetwork.org/v1/term/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sdmx-measure: <http://purl.org/linked-data/sdmx/2009/measure#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX taxon: <http://purl.uniprot.org/taxonomy/>
PREFIX xkos: <http://rdf-vocabulary.ddialliance.org/xkos#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX pubmed: <http://rdf.ncbi.nlm.nih.gov/pubmed/>
PREFIX schema: <https://schema.org/>
"""

# Compact ontology hints extracted from gn-transform-databases/examples/*.scm
_QUERY_HINTS = """ONTOLOGY
Classes: gnc:set(strain group),gnc:species,gnc:strain,dcat:Dataset,gnc:phenotype_trait,gnc:phenotype,gnc:gene,gnc:probeset,gnc:marker
data: dct:title,rdfs:label,skos:definition,gnt:gene_symbol,gnt:symbol,gnt:has_target_id,gnt:chr,gnt:mb
Links: gnt:has_trait_page→trait URL,gnt:has_phenotype_data→set→dataset,gnt:has_phenotype_trait→dataset→trait,gnt:has_strain→species→set,gnt:has_species→set→species,gnt:has_genotype_data→set→dataset
trait: gnt:has_trait_page→https://genenetwork.org/show_trait?trait_id=ID&dataset=NAME
PREFIX pubmed:→http://rdf.ncbi.nlm.nih.gov/pubmed/; taxon:→http://purl.uniprot.org/taxonomy/
EXAMPLES
1) Traits by keyword:
SELECT?t?name?page{?d a dcat:Dataset;gnt:has_phenotype_trait?t.?t rdfs:label?name;gnt:has_trait_page?page.FILTER(CONTAINS(LCASE(STR(?name)),LCASE("WORD")))}
2) Datasets by set:
SELECT?d?title{?s a gnc:set;rdfs:label?l;gnt:has_phenotype_data?d.?d a dcat:Dataset;dct:title?title.FILTER(CONTAINS(LCASE(STR(?l)),LCASE("SET")))}
3) Genes by symbol:
SELECT?g?sym{?g a gnc:gene;gnt:gene_symbol?sym.FILTER(CONTAINS(LCASE(STR(?sym)),LCASE("SYM")))}
4) Traits in dataset:
SELECT?t?name?page{?d a dcat:Dataset;dct:title?dt;gnt:has_phenotype_trait?t.?t rdfs:label?name;gnt:has_trait_page?page.FILTER(CONTAINS(LCASE(STR(?dt)),LCASE("DATASET")))}
RULES
- Every query starts with PREFIX block above.
- Only use declared prefixes. Do NOT invent new ones.
- When you see pubmed:ID in results, expand to http://rdf.ncbi.nlm.nih.gov/pubmed/ID.
- Do NOT prepend http://rdf.genenetwork.org/v1/id/ to URIs that are already complete.
- Use gnt:has_trait_page for trait URLs. Never construct trait URLs manually.
"""


class QueryTranslation(dspy.Signature):
    """Translate natural language to SPARQL SELECT. CRITICAL: every query MUST start with the PREFIX declarations. Only use declared prefixes. Use ontology hints and example patterns."""

    original_query: str = dspy.InputField(desc="User query")
    ontology_hints: str = dspy.InputField(desc="GeneNetwork ontology and example query patterns")
    translated_query: str = dspy.OutputField(
        desc="Valid SPARQL SELECT query with PREFIX declarations"
    )


_translate_query = dspy.Predict(QueryTranslation)


def sparql_fetch(query: str, sparql_uri: str) -> Any:
    sparql_query = _translate_query(
        original_query=query,
        ontology_hints=_SPARQL_PREFIXES + "\n" + _QUERY_HINTS,
    ).get("translated_query")
    sparql = SPARQLWrapper(sparql_uri)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(sparql_query)
    return sparql.queryAndConvert()


def make_sparql_fetch_tool(sparql_uri: str) -> dspy.Tool:
    def _fetch(query_str: str) -> Any:
        return sparql_fetch(query_str, sparql_uri)

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
