"""Utility for extracting pubmed papers"""

import json

from SPARQLWrapper import SPARQLWrapper, JSON


sparql = SPARQLWrapper("http://sparql-test.genenetwork.org/sparql/")
sparql.setReturnFormat(JSON)

query = """
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX fabio: <http://purl.org/spar/fabio/>

SELECT ?pubmedid
WHERE {
?paper rdf:type fabio:ResearchPaper .
?paper fabio:hasPubMedId ?pubmedid .
?paper dct:title ?title .
}
"""

sparql.setQuery(query)

results = sparql.queryAndConvert()

pubmed_ids = [result["pubmedid"]["value"] for result in results["results"]["bindings"]]

with open("pubmed_papers.txt", "w") as file:
    file.write(json.dumps(pubmed_ids))
