"""Common utilities"""

from SPARQLWrapper import SPARQLWrapper, JSON


def fetch_schema(endpoint_url: str):
    """Fetch schema (classes and properties) and return as rdflib Graph"""
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    # Fetch classes
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
    results = sparql.queryAndConvert()
    classes = results["results"]["bindings"]

    # Fetch properties
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
    results = sparql.queryAndConvert()
    properties = results["results"]["bindings"]

    return classes, properties
