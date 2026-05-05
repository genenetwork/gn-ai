"""Agent with SPARQL tool calling for AI search in GeneNetwork"""

from typing import Any

import dspy
from gnais.search.tools import (
    MemoryTools,
    check_link,
    make_sparql_fetch_tool,
    with_memory,
)
from gnais.search.prompts import GENERAL_SYSTEM_PROMPT


class AgentSig(dspy.Signature):
    query: str = dspy.InputField()
    chat_history: str = dspy.InputField()
    solution: str = dspy.OutputField(
        desc="""Return the answer as valid HTML only. Use only safe body tags such as
        <p>, <ul>, <ol>, <li>, <a>, <strong>, <em>, <code>, <pre>, and <br>.
        Do not use Markdown. Do not wrap the answer in ```html fences.

        Link correctness is mandatory. Every <a href="..."> URL MUST come directly
        from a SPARQL tool result returned by the configured Virtuoso endpoint.
        Never invent, infer, concatenate, normalize, rewrite, shorten, or guess URLs.
        Never create links from literals, labels, names, descriptions, IDs, QNames,
        CURIEs, blank nodes, or user text. Only bind and use URI/IRI values returned
        by SPARQL variables.

        Before emitting any hyperlink, verify that the href value is exactly one of
        the URI/IRI strings present in the latest relevant SPARQL result set. If no
        verified URI/IRI is available, omit the hyperlink and render plain text instead.
        Broken links are worse than no links.

        When querying SPARQL, prefer fast, selective, bounded queries. Use LIMIT,
        specific predicates, VALUES, FILTERs, and focused graph patterns where possible.
        Avoid broad scans, unbounded OPTIONAL chains, SELECT *, and expensive regex
        unless strictly necessary. Retrieve only the variables needed for the answer,
        especially URI/IRI variables intended for links.

        The final answer must include detailed reasoning where useful, followed by a
        clear final answer, but all content must remain valid HTML."""
    )


_STREAM_REACT_CACHE: dict[str, Any] = {}


def _get_stream_react(sparql_url: str) -> Any:
    """Return a cached (or freshly built) streaming ReAct agent."""
    if sparql_url not in _STREAM_REACT_CACHE:
        tools = [make_sparql_fetch_tool(sparql_url), check_link]
        _STREAM_REACT_CACHE[sparql_url] = dspy.streamify(
            dspy.ReAct(
                signature=AgentSig,
                tools=tools,
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
    return _STREAM_REACT_CACHE[sparql_url]


@with_memory(memory_type="agent")
async def agent_search(
    query: str,
    sparql_url: str,
    system_prompt: str = GENERAL_SYSTEM_PROMPT,
    user_id: str = "default_user",
    memory=None,
    chat_history: list = [],
):
    """Run agent-based search with SPARQL tool calling and optional memory.

    Yields stream chunks and a final prediction dict.
    """
    yield {"status": "Planning search strategy…"}
    stream_react = _get_stream_react(sparql_url)

    async for value in stream_react(
        query=f"{system_prompt}\nQuery: {query}",
        chat_history=chat_history,
    ):
        if isinstance(value, dspy.Prediction):
            solution = getattr(value, "solution", None)
            if solution:
                yield {"final": solution}
        else:
            yield getattr(value, "chunk", str(value))
