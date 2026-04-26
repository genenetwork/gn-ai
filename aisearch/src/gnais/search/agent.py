"""Agent with SPARQL tool calling for AI search in GeneNetwork"""

import os
from typing import Any

import dspy
from gnais.search.tools import make_sparql_fetch_tool


_SYSTEM_PROMPT = """Answer from SPARQL results. Work with partial data; do not apologize for query errors.
Links: expand ALL turtle prefixes before using in <a href>.
EXamples (not complete): pubmed:→http://rdf.ncbi.nlm.nih.gov/pubmed/ taxon:→http://purl.uniprot.org/taxonomy/
gn:→http://rdf.genenetwork.org/v1/id gnc:→http://rdf.genenetwork.org/v1/category gnt:→http://rdf.genenetwork.org/v1/term dcat:→http://www.w3.org/ns/dcat dct:→http://purl.org/dc/terms rdfs:→http://www.w3.org/2000/01/rdf-schema skos:→http://www.w3.org/2004/02/skos/core
Trait links: use the URL from gnt:has_trait_page. Never build trait URLs manually.
Format as HTML using <p>,<ul>,<li>,<a>,<strong>,<em>,<br>. No markdown blocks.
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
