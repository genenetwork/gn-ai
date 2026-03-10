#! /usr/bin/env python3
"""
This scripts builds documents that we use for our RAG model.
"""

import json
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON
from string import Template

PREFIXES = {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "ex": "http://example.org/stuff/1.0/",
    "fabio": "http://purl.org/spar/fabio/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "generif": "http://www.ncbi.nlm.nih.gov/gene?cmd=Retrieve&dopt=Graphics&list_uids=",
    "gn": "http://rdf.genenetwork.org/v1/id/",
    "gnc": "http://rdf.genenetwork.org/v1/category/",
    "gnt": "http://rdf.genenetwork.org/v1/term/",
    "owl": "http://www.w3.org/2002/07/owl#",
    "phenotype": "http://rdf.genenetwork.org/v1/phenotype/",
    "prism": "http://prismstandard.org/namespaces/basic/2.0/",
    "publication": "http://rdf.genenetwork.org/v1/publication/",
    "pubmed": "http://rdf.ncbi.nlm.nih.gov/pubmed/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "taxon": "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=",
    "up": "http://purl.uniprot.org/core/",
    "xkos": "http://rdf-vocabulary.ddialliance.org/xkos#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

RDF_PREFIXES = "\n".join(
    [f"PREFIX {key}: <{value}>" for key, value in PREFIXES.items()]
)

DEFAULT_RESOURCES = [
    "gnc:resource_classification_scheme",
    "gnc:set",
    "gnc:taxonomic_family"
]

OUTPUT_JSON_FILE = "metadata_documents.json"
APPEND_OUTPUT = False
ret = {}

PREFERRED_SUMMARY_PREDICATES = (
    "skos:definition",
    "dct:description",
    "rdfs:comment",
)

PREFERRED_TITLE_PREDICATES = (
    "skos:prefLabel",
    "rdfs:label",
    "dct:title",
    "foaf:name",
)


def _uri_to_prefixed(uri: str) -> str:
    for prefix, namespace in PREFIXES.items():
        if uri.startswith(namespace):
            return f"{prefix}:{uri[len(namespace):]}"
    return uri


def _resource_to_sparql_term(resource: str) -> str:
    if resource.startswith("<") and resource.endswith(">"):
        return resource
    if resource.startswith("http://") or resource.startswith("https://"):
        return f"<{resource}>"
    return resource


def _binding_value_to_text(binding_value: dict) -> str:
    value_type = binding_value.get("type")
    value = binding_value.get("value", "")
    if value_type == "uri":
        compact = _uri_to_prefixed(value)
        return compact if compact != value else value
    return value


def _merge_unique_resources(base: list[str], extra: list[str]) -> list[str]:
    merged = list(base)
    seen = set(base)
    for resource in extra:
        if resource not in seen:
            merged.append(resource)
            seen.add(resource)
    return merged


def _humanize_predicate(predicate: str) -> str:
    name = predicate.split(":", 1)[-1]
    words = []
    current = ""
    for char in name:
        if char == "_":
            if current:
                words.append(current)
                current = ""
            continue
        if char.isupper() and current:
            words.append(current)
            current = char.lower()
        else:
            current += char
    if current:
        words.append(current)
    return " ".join(words).strip().capitalize()


def _build_grouped_bindings(query_result: dict) -> dict[str, list[str]]:
    bindings = query_result.get("results", {}).get("bindings", [])
    grouped = defaultdict(list)
    seen = defaultdict(set)
    for item in bindings:
        predicate = _binding_value_to_text(item.get("p", {}))
        object_value = _binding_value_to_text(item.get("o", {})).strip()
        key = object_value.casefold()
        if object_value and key not in seen[predicate]:
            grouped[predicate].append(object_value)
            seen[predicate].add(key)
    return grouped


def _build_document_text(query_result: dict, subject: str) -> str:
    grouped = _build_grouped_bindings(query_result)
    if not grouped:
        return "\n".join(
            [
                f"Resource: {subject}",
                "Summary: No metadata triples were returned for this resource.",
            ]
        )

    title = subject
    for predicate in PREFERRED_TITLE_PREDICATES:
        values = grouped.get(predicate)
        if values:
            title = values[0]
            break

    summary = None
    for predicate in PREFERRED_SUMMARY_PREDICATES:
        values = grouped.get(predicate)
        if values:
            summary = values[0]
            break

    triple_count = sum(len(values) for values in grouped.values())
    types = grouped.get("rdf:type", [])
    number_of_levels = None
    identifier = None

    for predicate, fact_key in (("xkos:numberOfLevels", "number_of_levels"), ("dct:identifier", "identifier")):
        values = grouped.get(predicate)
        if values:
            value = values[0] if len(values) == 1 else ", ".join(values)
            if fact_key == "number_of_levels":
                number_of_levels = value
            else:
                identifier = value

    narrative_lines = [f"Title: {title}", f"Resource: {subject}"]
    if summary:
        narrative_lines.append(f"Summary: {summary}")
    narrative_lines.append("")
    narrative_lines.append("Key facts:")
    narrative_lines.append(f"- Triple count: {triple_count}")
    if types:
        narrative_lines.append(f"- Type: {', '.join(types)}")
    if number_of_levels:
        narrative_lines.append(f"- Number of levels: {number_of_levels}")
    if identifier:
        narrative_lines.append(f"- Identifier: {identifier}")
    narrative_lines.append("")
    narrative_lines.append("Attributes:")

    excluded = set(PREFERRED_TITLE_PREDICATES) | set(PREFERRED_SUMMARY_PREDICATES) | {
        "rdf:type",
        "xkos:numberOfLevels",
        "dct:identifier",
    }
    for predicate, values in sorted(grouped.items()):
        if predicate in excluded:
            continue
        label = _humanize_predicate(predicate)
        if len(values) == 1:
            narrative_lines.append(f"- {label} [{predicate}]: {values[0]}")
        else:
            narrative_lines.append(f"- {label} [{predicate}]: {', '.join(values)}")

    return "\n".join(narrative_lines)


def fetch_resource_metadata(sparql: SPARQLWrapper, resource: str) -> dict:
    resource_term = _resource_to_sparql_term(resource)
    query = Template("""
$prefix

SELECT ?p ?o
FROM <http://rdf.genenetwork.org/v1>
WHERE {
  $resource ?p ?o .
}
ORDER BY ?p ?o
""").substitute(prefix=RDF_PREFIXES, resource=resource_term)
    sparql.setQuery(query)
    return sparql.queryAndConvert()


def fetch_reference_population_resources(sparql: SPARQLWrapper) -> list[str]:
    query = Template("""
$prefix

SELECT ?resource
FROM <http://rdf.genenetwork.org/v1>
WHERE {
  gnc:population_category gnt:has_reference_population ?resource .
}
""").substitute(prefix=RDF_PREFIXES)
    sparql.setQuery(query)
    result = sparql.queryAndConvert()
    bindings = result.get("results", {}).get("bindings", [])
    resources = []
    for item in bindings:
        value = item.get("resource", {}).get("value", "")
        if value:
            resources.append(_uri_to_prefixed(value))
    return resources


def fetch_species_resources(sparql: SPARQLWrapper) -> list[str]:
    query = Template("""
$prefix

SELECT ?resource
FROM <http://rdf.genenetwork.org/v1>
WHERE {
  gnc:species skos:member ?resource .
}
""").substitute(prefix=RDF_PREFIXES)
    sparql.setQuery(query)
    result = sparql.queryAndConvert()
    bindings = result.get("results", {}).get("bindings", [])
    resources = []
    for item in bindings:
        value = item.get("resource", {}).get("value", "")
        if value:
            resources.append(_uri_to_prefixed(value))
    return resources


def fetch_taxonomic_family_resources(sparql: SPARQLWrapper) -> list[str]:
    query = Template("""
$prefix

SELECT ?resource
FROM <http://rdf.genenetwork.org/v1>
WHERE {
  gnc:taxonomic_family gnt:has_taxonomic_family ?resource .
}
""").substitute(prefix=RDF_PREFIXES)
    sparql.setQuery(query)
    result = sparql.queryAndConvert()
    bindings = result.get("results", {}).get("bindings", [])
    resources = []
    for item in bindings:
        value = item.get("resource", {}).get("value", "")
        if value:
            resources.append(_uri_to_prefixed(value))
    return resources


def fetch_set_resources(sparql: SPARQLWrapper) -> list[str]:
    query = Template("""
$prefix

SELECT ?resource
FROM <http://rdf.genenetwork.org/v1>
WHERE {
  gnc:set skos:member ?resource .
}
""").substitute(prefix=RDF_PREFIXES)
    sparql.setQuery(query)
    result = sparql.queryAndConvert()
    bindings = result.get("results", {}).get("bindings", [])
    resources = []
    for item in bindings:
        value = item.get("resource", {}).get("value", "")
        if value:
            resources.append(_uri_to_prefixed(value))
    return resources


def build_documents(resources: list[str]) -> tuple[list[str], dict]:
    docs = []
    per_resource_ret = {}
    sparql = SPARQLWrapper("https://rdf.genenetwork.org/sparql")
    sparql.setReturnFormat(JSON)

    for resource in resources:
        ret = fetch_resource_metadata(sparql, resource)
        per_resource_ret[resource] = ret
        docs.append(_build_document_text(ret, subject=resource))

    return docs, per_resource_ret


if __name__ == "__main__":
    discovery_sparql = SPARQLWrapper("https://rdf.genenetwork.org/sparql")
    discovery_sparql.setReturnFormat(JSON)
    reference_populations = fetch_reference_population_resources(discovery_sparql)
    species_resources = fetch_species_resources(discovery_sparql)
    taxonomic_family_resources = fetch_taxonomic_family_resources(discovery_sparql)
    set_resources = fetch_set_resources(discovery_sparql)
    resources = _merge_unique_resources(DEFAULT_RESOURCES, reference_populations)
    resources = _merge_unique_resources(resources, species_resources)
    resources = _merge_unique_resources(resources, taxonomic_family_resources)
    resources = _merge_unique_resources(resources, set_resources)

    docs, ret_by_resource = build_documents(resources)

    # Preserve `ret` for debugging/inspection workflows.
    ret = ret_by_resource[resources[0]] if resources else {}

    output_payload = docs
    if APPEND_OUTPUT:
        try:
            with open(OUTPUT_JSON_FILE, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if not isinstance(existing, list):
                raise ValueError(f"Expected JSON array in {OUTPUT_JSON_FILE}")
        except FileNotFoundError:
            existing = []
        output_payload = existing + docs
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # Validate output file is valid JSON before finishing.
    with open(OUTPUT_JSON_FILE, "r", encoding="utf-8") as handle:
        json.load(handle)

    print(json.dumps(docs, indent=2, ensure_ascii=False))
