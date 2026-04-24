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

    # Fetch examples
    example_query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT (SAMPLE(?subject) AS ?subject) ?predicate ?object
    WHERE {
    {
    { ?subject a rdfs:Class }
    UNION { ?subject a owl:Class }
    UNION { ?subject rdfs:domain ?object }
    UNION { ?subject rdfs:range ?object }
    UNION { ?subject rdfs:subClassOf ?object }
    UNION { ?subject a owl:ObjectProperty }
    UNION { ?subject a owl:DatatypeProperty }
    ?subject ?predicate ?object
    FILTER (?predicate != skos:member)
    }
    UNION
    {
    SELECT ?subject ?predicate (SAMPLE(?obj) AS ?object)
    WHERE {
    ?subject skos:member ?obj .
    BIND(skos:member AS ?predicate)
    BIND(LCASE(REPLACE(STR(?obj), "^([^_]*_[^_]*_).*$", "$1")) AS ?stem)
    FILTER (?subject != ?obj)
    }
    GROUP BY ?subject ?predicate ?stem
    }
    }
    GROUP BY ?predicate ?object
    ORDER BY ?predicate ?object
    """
    sparql.setQuery(example_query)
    results = sparql.queryAndConvert()
    examples = results["results"]["bindings"]
    
    return classes, properties, examples
