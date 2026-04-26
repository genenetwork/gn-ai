"""Main module of hybrid search for GeneNetwork"""

import os
import argparse
import asyncio
import dspy

import torch
from rich.console import Console
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from gnais.search.ragent import hybrid_search

SEED = os.getenv("SEED")
if SEED is None:
    raise ValueError("SEED must be specified for reproducibility")
MODEL_NAME = os.getenv("MODEL_NAME")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME must be specified - either proprietary or local")
MODEL_TYPE = os.getenv("MODEL_TYPE")
if MODEL_TYPE is None:
    raise ValueError("MODEL_TYPE must be specified")


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


if int(MODEL_TYPE) == 0:
    llm = dspy.LM(
        model=f"openai/{MODEL_NAME}",
        api_base="http://localhost:7501/v1",
        api_key="local",
        model_type="chat",
        max_tokens=10_000,
        n_ctx=10_000,
        seed=2_025,
        temperature=0,
        verbose=False,
    )
elif int(MODEL_TYPE) == 1:
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Valid API_KEY must be specified to use the proprietary model")
    llm = dspy.LM(
        MODEL_NAME,
        api_key=API_KEY,
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
else:
    raise ValueError("MODEL_TYPE must be 0 or 1")


dspy.configure(lm=llm)


class _HybridDisplay:
    """Terminal UI with three live panels for RAG / GRAG / Agent."""

    _COLORS = {
        "rag": "cyan",
        "grag": "green",
        "agent": "yellow",
        "synthesis": "magenta",
    }

    def __init__(self):
        self._texts = {
            "rag": Text("", end=""),
            "grag": Text("", end=""),
            "agent": Text("", end=""),
            "synthesis": Text("", end=""),
        }
        self._status = {
            "rag": "⏳ waiting",
            "grag": "⏳ waiting",
            "agent": "⏳ waiting",
            "synthesis": "⏳ waiting",
        }
        self._live = Live(self._render(), refresh_per_second=10)

    def _render(self):
        panels = []
        for src in ("rag", "grag", "agent", "synthesis"):
            color = self._COLORS[src]
            title = f"[bold {color}]{src.upper()}[/] — {self._status[src]}"
            panels.append(
                Panel(
                    self._texts[src],
                    title=title,
                    border_style=color,
                    expand=True,
                )
            )
        return Columns(panels, equal=True)

    def start(self):
        self._live.start()

    def stop(self):
        self._live.stop()

    def append(self, source: str, text: str):
        if source in self._texts:
            self._texts[source].append(text)
        self._live.update(self._render())

    def set_status(self, source: str, status: str):
        if source in self._status:
            self._status[source] = status
        self._live.update(self._render())


async def digest(query: str):
    display = _HybridDisplay()
    display.start()

    final_html = None
    try:
        async for event in hybrid_search(query):
            source = event["source"]
            kind = event["kind"]
            content = event["content"]

            if source == "hybrid" and kind == "final":
                final_html = content
                break

            if source == "synthesis" and kind == "chunk":
                display.set_status(source, "🔄 synthesizing")
                display.append(source, content)
                continue

            if kind == "chunk":
                display.append(source, content)
            elif kind == "final":
                display.set_status(source, "✅ complete")
                display.append(source, content)
            elif kind == "error":
                display.set_status(source, f"❌ error")
                display.append(source, f"\n[ERROR] {content}\n")
            elif kind == "done":
                display.set_status(source, "✅ done")
    finally:
        display.stop()

    console = Console()
    console.print()
    if final_html:
        console.rule("[bold magenta]HYBRID SYNTHESIS[/bold magenta]")
        console.print(final_html)
    else:
        console.print("[bold red]No final synthesis received.[/bold red]")
    console.print()
    return final_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    args = parser.parse_args()

    asyncio.run(digest(args.query))
