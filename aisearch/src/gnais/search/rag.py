"""
Module with RAG system for AI search in GeneNetwork
Embedding model = Qwen/Qwen3-Embedding-0.6B
"""

__all__ = (
    "AISearch",
)

import json
import uuid
from copy import deepcopy
from typing import Any

from chromadb.config import Settings
import dspy
from gnais.search.config import generate, reformat, generate_stream
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class RAGSearch:
    """
    Represent RAG system used for AI search
    """
    def __init__(self, ensemble_retriever: Any, stream: bool = False):
        self.stream = stream
        self.ensemble_retriever = ensemble_retriever
        # Init the ensemble retriever
        graph_builder = StateGraph(State)
        graph_builder.add_node("chat", self.chat)
        graph_builder.add_edge(START, "chat")
        graph_builder.add_edge("chat", END)
        self.graph = graph_builder.compile()
        self.stream_predict = dspy.streamify(
            generate_stream,
            stream_listeners=[
                dspy.streaming.StreamListener(
                    signature_field_name="feedback", allow_reuse=True
                )
            ],
            include_final_prediction_in_output_stream=True,
        )

    def chat(self, state: State) -> dict:
        """Run user query through the RAG system for search

        Args:
            query: user query

        Returns:
            response to user query
        """
        predictor = generate_stream if self.stream else generate
        kwargs = {"config": {"cache": False}} if self.stream else {}
        return self._chat_response(state, predictor, **kwargs)

    def _prepare_generation_inputs(
        self, query: str, chat_history: list[BaseMessage]
    ) -> dict[str, Any]:
        retrieved_docs = self.ensemble_retriever.invoke(query)
        return {
            "input_text": f"""
You excel at addressing search query using the context and chat history you have. You do not make mistakes.
Extract answers to the query below from the context and chat history. Use the chat history before moving to the context.
Provide links associated with each RDF entity. To build links you must replace RDF prefixes by namespaces.
Here is the mapping of prefixes and namespaces:
gn => http://rdf.genenetwork.org/v1/id
gnc => http://rdf.genenetwork.org/v1/category
owl => http://www.w3.org/2002/07/owl
gnt => http://rdf.genenetwork.org/v1/term
skos = http://www.w3.org/2004/02/skos/core
xkos => http://rdf-vocabulary.ddialliance.org/xkos
rdf => http://www.w3.org/1999/02/22-rdf-syntax-ns
rdfs => http://www.w3.org/2000/01/rdf-schema
taxon => http://purl.uniprot.org/taxonomy
dcat => http://www.w3.org/ns/dcat
dct => http://purl.org/dc/terms
xsd => http://www.w3.org/2001/XMLSchema
sdmx-measure => http://purl.org/linked-data/sdmx/2009/measure
qb => http://purl.org/linked-data/cube
pubmed => http://rdf.ncbi.nlm.nih.gov/pubmed
v => http://www.w3.org/2006/vcard/ns
foaf => http://xmlns.com/foaf/0.1
geoSeries => http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc

All links pointing to specific traits should be translated to CD links using the trait id and the dataset name.
Original trait link: https://rdf.genenetwork.org/v1/id/trait_16339
Trait id: 16339
Dataset name: BXDPublish
New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish\n
Format your entire response as valid HTML. Use tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>. Do not wrap the response in markdown code blocks.

{query}""",
            "chat_history": chat_history,
            "context": retrieved_docs,
        }

    def _state_to_generation_inputs(self, state: State) -> dict[str, Any]:
        messages = state.get("messages")
        query = deepcopy(messages[-1].content)
        chat_history = deepcopy(messages)
        return self._prepare_generation_inputs(query, chat_history)

    def _prediction_feedback(self, prediction: Any) -> str:
        return str(prediction.get("feedback"))

    def _chat_response(self, state: State, predictor: Any, **kwargs) -> dict:
        response = predictor(**self._state_to_generation_inputs(state), **kwargs)
        return {"messages": [self._prediction_feedback(response)]}

    async def call_graph(self, query: str) -> Any:
        config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return result

    async def _handle(self, query: str) -> str:
        result = await self.call_graph(query)
        result = result.get("messages")[-1].content
        reformatted = reformat(input_text=result).get(
            "result"
        )  # transform to valid nested dictionary
        return reformatted

    async def _handle_stream(self, query: str):
        """Yield incremental text updates for a query."""
        output = self.stream_predict(
            **self._prepare_generation_inputs(
                query,
                [HumanMessage(content=query)],
            ),
            config={"cache": False},
        )
        async for value in output:
            if hasattr(value, "chunk"):
                yield value.chunk
            elif isinstance(value, dspy.Prediction):
                feedback = self._prediction_feedback(value)
                if feedback:
                    yield {"final": feedback}

    def handle(self, query: str) -> Any:
        """Handle a query with optional streaming depending on ``self.stream``."""

        if self.stream:
            return self._handle_stream(query)
        return self._handle(query)
