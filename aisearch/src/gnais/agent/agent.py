"""Agent with sparql tool calling for AI search in GeneNetwork"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
from gnais.rag.config import ListInformation, reformat
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from SPARQLWrapper import JSON, SPARQLWrapper
from typing_extensions import Annotated, TypedDict

THREAD = uuid.uuid4().hex[:8]

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
    if Path(f"{TTL_PATH}/schema.txt").exists():
        with open(f"{TTL_PATH}/schema.txt") as f:
            prefixes, predicates = json.loads(f.read())
    else:
        prefixes, predicates = extract_schema(TTL_PATH)
        with open(f"{TTL_PATH}/schema.txt", "w") as f:
            f.write(json.dumps([prefixes, predicates]))
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


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass
class Digest:
    memory: Any = field(init=False)
    graph: Any = field(init=False)

    def __post_init__(self):
        self.memory = MemorySaver()
        graph_builder = StateGraph(State)
        graph_builder.add_node("chat", self.chat)
        graph_builder.add_edge(START, "chat")
        graph_builder.add_edge("chat", END)
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def chat(self, state: State) -> dict:
        query = state.get("messages")
        system_prompt = """
            You excel at addressing search query using the context you have. You do not make mistakes.
            Extract answers to the query from the context and provide links associated with each RDF entity.
            All links pointing to specific traits should be translated to CD links using the trait id (numeric code) and the dataset name specifically.
            Original trait link: https://rdf.genenetwork.org/v1/id/trait_BXD_16339
            Trait id: 16339
            Dataset name: BXDPublish
            New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish\n
            """
        final_query = system_prompt + str(query)
        search = AISearch()
        output = search(query=final_query).get("solution")
        output = str(output)
        return {"messages": [output]}

    async def call_graph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": THREAD}}
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    def handle(self, query: str) -> str:
        result = asyncio.run(self.call_graph(query))
        result = result.get("messages")[-1].content
        reformatted = reformat(input_text=result).get(
            "result"
        )  # transform to valid nested dictionary
        return reformatted
