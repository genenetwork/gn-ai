"""Agent with SPARQL tool calling for AI search in GeneNetwork"""

from typing import Any

import dspy
from gnais.search.prompts import GENERAL_SYSTEM_PROMPT
from gnais.search.tools import (
    MemoryTools,
    check_link,
    make_sparql_fetch_tool,
    route_model,
    with_memory,
)


class AgentSig(dspy.Signature):
    query: str = dspy.InputField()
    chat_history: list = dspy.InputField(desc="Previous conversation context")
    solution: str = dspy.OutputField(
        desc="""Return the answer as valid HTML only. Use the chat
history as context.  Use only safe body tags such as <p>, <ul>, <ol>,
<li>, <a>, <strong>, <em>, <code>, <pre>, and <br>.  Do not use
Markdown. Do not wrap the answer in ```html fences.

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

The final answer must include detailed reasoning where useful, followed by a
clear final answer, but all content must remain valid HTML."""
    )


def _make_agent_stream(sparql_url: str):
    routed = route_model()(dspy.Predict(AgentSig))
    chosen_model = routed.choose_model()
    chosen_lm = routed.options[chosen_model]
    print(f"Choice made: {chosen_model} for Agent")
    tools = [
        make_sparql_fetch_tool(sparql_url, chosen_lm),
        check_link,
    ]
    react = dspy.ReAct(
        signature=AgentSig,
        tools=tools,
        max_iters=7,
    )
    react.set_lm(chosen_lm)
    return dspy.streamify(
        react,
        stream_listeners=[
            dspy.streaming.StreamListener(
                signature_field_name="next_thought",
                allow_reuse=True,
            ),
            dspy.streaming.StreamListener(signature_field_name="solution"),
        ],
        include_final_prediction_in_output_stream=True,
    )


@with_memory(memory_type="agent")
async def agent_search(
    query: str,
    sparql_url: str,
    system_prompt: str = GENERAL_SYSTEM_PROMPT,
    user_id: str = "default_user",
    memory=None,
    chat_history: list = [],
):
    yield {"status": "Planning search strategy…"}
    yield {"status": "Streaming response…"}
    async for value in _make_agent_stream(sparql_url)(
        query=f"{system_prompt}\nQuery: {query}",
        chat_history=chat_history,
    ):
        if isinstance(value, dspy.Prediction):
            yield {"final": value.solution}
        else:
            yield value.chunk
