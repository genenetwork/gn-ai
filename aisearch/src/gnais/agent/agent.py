"""Agent with sparql tool calling for AI search in GeneNetwork"""

import os
from typing import Any

import dspy
from gnais.rag.config import ListInformation
from SPARQLWrapper import JSON, SPARQLWrapper

TTL_PATH = os.getenv("TTL_PATH")
if TTL_PATH is None:
    raise FileNotFoundError(
        "TTL_PATH must be specified to extract RDF schema and build queries"
    )


def extract_schema(ttl_path: str) -> tuple[list, list]:
    ttl_files = [
        os.path.join(ttl_path, ttl) for ttl in os.listdir(ttl_path) if "ttl" in ttl
    ]
    prefixes = []
    predicates = []
    for ttl in ttl_files:
        with open(ttl) as f:
            contents = f.readlines()
            for content in contents:
                content = content.strip()
                if content.startswith("@") and content not in prefixes:
                    prefixes.append(content)
                elif len(content) != 0:
                    predicate = content.split()[1]
                    if predicate not in predicates:
                        predicates.append(predicate)
    return prefixes, predicates


class QueryTranslation(dspy.Signature):
    original_query: str = dspy.InputField()
    rdf_prefixes: list[str] = dspy.InputField()
    triple_predicates: list[str] = dspy.InputField()
    translated_query: str = dspy.OutputField(
        desc="SPARQL query corresponding to user query for fetching requested data given RDF schema inferred from RDF prefixes and triple predicates"
    )


translate_query = dspy.Predict(QueryTranslation)


def fetch_data(query: str) -> Any:
    sparql = SPARQLWrapper("https://rdf.genenetwork.org/sparql/")
    sparql.setReturnFormat(JSON)
    prefixes, predicates = extract_schema(TTL_PATH)
    sparql_query = translate_query(
        original_query=query, rdf_prefixes=prefixes, triple_predicates=predicates
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
            max_iters=20,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: str):
        return self.react(query=query)
