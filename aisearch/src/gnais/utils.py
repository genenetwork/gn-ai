"""Common utilities"""

from SPARQLWrapper import SPARQLWrapper, JSON

def fetch_schema(endpoint_url: str):
    """Fetch schema (classes and properties) and return as rdflib Graph"""
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    # Fetch classes
    schema_query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?subject ?predicate ?object
    WHERE {
    { ?subject a rdfs:Class }
    UNION { ?subject a owl:Class }
    UNION { ?subject rdfs:domain ?object }
    UNION { ?subject rdfs:range ?object }
    UNION { ?subject rdfs:subClassOf ?object }
    UNION { ?subject a owl:ObjectProperty }
    UNION { ?subject a owl:DatatypeProperty }
    ?subject ?predicate ?object
    }
    LIMIT 5000
    """
    sparql.setQuery(schema_query)
    results = sparql.queryAndConvert()
    schema = results["results"]["bindings"]
    
    return schema
