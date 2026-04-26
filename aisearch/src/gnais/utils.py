"""Common utilities"""

from SPARQLWrapper import SPARQLWrapper, JSON


def fetch_schema(endpoint_url: str):
    """Fetch schema (classes and properties) and return as clean strings for LLM consumption."""
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    classes_query = """
    SELECT DISTINCT ?class ?label ?comment
    WHERE {
        { ?class a owl:Class }
        UNION
        { ?class a rdfs:Class }
        OPTIONAL { ?class rdfs:label ?label }
        OPTIONAL { ?class rdfs:comment ?comment }
    }
    """
    sparql.setQuery(classes_query)
    classes = sparql.queryAndConvert()["results"]["bindings"]

    properties_query = """
    SELECT DISTINCT ?prop ?domain ?range ?label
    WHERE {
        { ?prop a owl:ObjectProperty }
        UNION
        { ?prop a owl:DatatypeProperty }
        UNION
        { ?prop a rdf:Property }
        OPTIONAL { ?prop rdfs:domain ?domain }
        OPTIONAL { ?prop rdfs:range ?range }
        OPTIONAL { ?prop rdfs:label ?label }
    }
    """
    sparql.setQuery(properties_query)
    properties = sparql.queryAndConvert()["results"]["bindings"]

    # Format classes for LLM
    lines = []
    for c in classes:
        uri = c.get("class", {}).get("value", "")
        label = c.get("label", {}).get("value", "")
        comment = c.get("comment", {}).get("value", "")
        parts = [uri]
        if label:
            parts.append(f"label={label!r}")
        if comment:
            parts.append(f"comment={comment!r}")
        lines.append("  " + " | ".join(parts))
    classes_str = "\n".join(lines) if lines else "  (no classes found)"

    # Format properties for LLM
    lines = []
    for p in properties:
        uri = p.get("prop", {}).get("value", "")
        label = p.get("label", {}).get("value", "")
        domain = p.get("domain", {}).get("value", "")
        range_ = p.get("range", {}).get("value", "")
        parts = [uri]
        if label:
            parts.append(f"label={label!r}")
        if domain:
            parts.append(f"domain={domain}")
        if range_:
            parts.append(f"range={range_}")
        lines.append("  " + " | ".join(parts))
    properties_str = "\n".join(lines) if lines else "  (no properties found)"

    return classes_str, properties_str
