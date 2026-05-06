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

When querying SPARQL, prefer fast, efficient SPARQL SELECT queries
that avoid Virtuoso timeouts (504 errors).

CRITICAL PERFORMANCE RULES (to prevent 504s):
1. Always add `LIMIT` - start with `LIMIT 50`, increase only if needed. Never omit `LIMIT`.
2. Never use `SELECT *` - list only the variables you actually need.
3. Avoid expensive operations: no Cartesian products, no cross joins, no full graph scans.
4. Use specific FILTER patterns that leverage indexes:
   - Prefer `STRSTARTS(?label, "prefix")` over `CONTAINS` or regex.
   - Avoid `FILTER regex(...)` - it disables indexes.
   - Use `FILTER(?value = "exact")` or `IN` with small lists.
5. Prefer property paths over multiple joins when traversing a chain.
6. Use VALUES blocks for small sets of constants instead of UNION or OPTIONAL.
7. Avoid ORDER BY on large result sets - if needed, combine with `LIMIT` and a narrow `WHERE` clause.
8. Never use nested subqueries unless absolutely necessary; flatten them.
9. Use `OPTIONAL` only for truly optional patterns – otherwise, use a simple triple pattern.
10. Limit the number of generated queries - output at most 10 (not 20) per request.

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
