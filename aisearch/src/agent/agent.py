"""Agent with sparql tool calling for AI search in GeneNetwork"""

__all__ = "AISearch"


import os
from typing import Any

import dspy
from gnais.rag.config import ListInformation
from SPARQLWrapper import SPARQLWrapper

TTL_PATH = os.getenv("TTL_PATH")
if TTL_PATH is None:
    raise FileNotFoundError(
        "TTL_PATH must be specified to extract RDF schema and build queries"
    )


def extract_schema(ttl_path: str) -> tuple[list, list]:
    ttl_files = [os.path.join(dir_path, ttl) for ttl in os.listdir(ttl_path)]
    prefixes = []
    properties = []
    for ttl in ttl_files:
        with open(ttl) as f:
            contents = f.readlines()
        for content in contents:
            if content.startswith("@") and content not in prefixes:
                prefixes.append(content)
            else:
                prop = content.split()[1]
                if prop not in properties:
                    properties.append(prop)
    return prefixes, properties


class Translation(dspy.signature):
    user_query: str = dspy.InputField()
    prefixes: list[str] = dspy.InputField("RDF prefixes for namespaces")
    properties: list[str] = dspy.InputField(
        "Properties linking subject and object in RDF triples"
    )
    translated_query: str = dspy.OutputField(
        desc="SPARQL query corresponding to the user query for fetching requested data given RDF properties and prefixes"
    )


translate = dspy.Predict(Translation)


def fetch_data(user_query: str) -> Any:
    sparql = SPARQLWrapper("https://rdf.genenetwork.org/sparql")
    prefixes, properties = extract_schema(TTL_PATH)
    sparql_query = translate(
        user_query=user_query, prefixes=prefixes, properties=properties
    ).get("translated_query")
    sparql.setQuery(sparql_query)
    return sparql.queryAndConvert()


fetch_data = dspy.Tool(
    name="fetch_data",
    desc="Fetch RDF data around GeneNetwork data through SPARQL",
    args={
        "query": {
            "type": "string",
            "desc": "SPARQL query to run to fetch relevant data",
        },
    },
    func=fetch_data,
)


class ReactSig(dspy.Signature):
    query: str = dspy.InputField()
    solution: ListInformation = dspy.OutputField(desc="The answer to the query")


class AISearch(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [fetch_data]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=50,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: str):
        return self.react(query=query)
