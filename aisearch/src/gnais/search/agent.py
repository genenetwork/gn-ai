"""Agent with SPARQL tool calling for AI search in GeneNetwork"""

import os
from typing import Any

import dspy
from gnais.search.tools import make_sparql_fetch_tool


_SYSTEM_PROMPT = """You excel at addressing search query using the context you have. You do not make mistakes.
Extract answers to the query from the context and provide links associated with each RDF entity.
All links pointing to specific traits should be translated to CD links using the trait id (numeric code) and the dataset name specifically.
Original trait link: https://rdf.genenetwork.org/v1/id/trait_BXD_16339
Trait id: 16339
Dataset name: BXDPublish
New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish

Format your entire response as valid HTML. Use tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>. Do not wrap the response in markdown code blocks.
"""


class AgentSig(dspy.Signature):
    query: str = dspy.InputField()
    solution: str = dspy.OutputField(
        desc="The answer to the query with detailed answers and the final answer, formatted as valid HTML using tags such as <p>, <ul>, <li>, <a>, <strong>, <em>, and <br>"
    )


async def agent_search(query: str, sparql_url: str, system_prompt: str = _SYSTEM_PROMPT, user_id: str = "default_user"):
    """Run agent-based search with SPARQL tool calling.

    Yields stream chunks and a final prediction dict.
    """
    stream_react = dspy.streamify(
        dspy.ReAct(
            signature=AgentSig,
            tools=[make_sparql_fetch_tool(sparql_url)],
            max_iters=20,
        ),
        stream_listeners=[
            dspy.streaming.StreamListener(
                signature_field_name="next_thought",
                allow_reuse=True,
            ),
            dspy.streaming.StreamListener(signature_field_name="solution"),
        ],
        include_final_prediction_in_output_stream=True,
    )

    async for value in stream_react(
            query=f"{_SYSTEM_PROMPT}\n{query}",
            config={"cache": False}
    ):
        if isinstance(value, dspy.Prediction):
            solution = getattr(value, "solution", None)
            if solution:
                yield {"final": solution}
        else:
            yield getattr(value, "chunk", str(value))
