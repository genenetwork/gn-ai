"""Main module of hybrid search for GeneNetwork"""

import argparse
import asyncio
import os

import dspy
import torch
from dotenv import load_dotenv
from gnais.search.prompts import GN_FACT_EXTRACTION_PROMPT, GN_UPDATE_MEMORY_PROMPT
from gnais.search.ragent import hybrid_search
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


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


def digest(query: str, user_id: str, memory: Memory = None):
    async def _run():
        display = _HybridDisplay()
        display.start()

        final_html = None
        try:
            async for event in hybrid_search(query, user_id=user_id, memory=memory):
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

    return asyncio.run(_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--user-id", default="test-user", help="Mem0 user identity")
    parser.add_argument("query", help="Search query")
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)

    DB_PATH = os.getenv("DB_PATH")
    SEED = int(os.getenv("SEED"))
    MODEL_NAME = os.getenv("MODEL_NAME")
    MODEL_TYPE = int(os.getenv("MODEL_TYPE"))
    API_KEY = os.getenv("API_KEY")
    PORT = os.getenv("PORT")
    
    torch.manual_seed(SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    llm = dspy.LM(
        model=MODEL_NAME if MODEL_TYPE else f"ollama_chat/{MODEL_NAME}",
        api_key=API_KEY if MODEL_TYPE else "local",
        api_base = None if MODEL_TYPE else f"http://localhost:{PORT}",
        max_tokens=100_000,
        temperature=0,
        verbose=False,
    )
    dspy.configure(lm=llm)

    os.environ[f"{MODEL_NAME.split('/')[0].upper()}_API_KEY"] = API_KEY
    memory_config = MemoryConfig(
        custom_fact_extraction_prompt=GN_FACT_EXTRACTION_PROMPT,
        custom_update_extraction_prompt=GN_UPDATE_MEMORY_PROMPT,
        llm={
            "provider": "litellm",
            "config": {
                "model": MODEL_NAME,
                "temperature": 0.1,
                "max_tokens": 2_000,
                "api_key": API_KEY,
            },
        },
        embedder={
            "provider": "huggingface",
            "config": {
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "embedding_dims": 1024,
                "model_kwargs": {
                    "trust_remote_code": True,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
            },
        },
        vector_store={
            "provider": "chroma",
            "config": {
                "collection_name": "mem0",
                "path": os.path.join(DB_PATH, "mem0_chroma"),
            },
        },
    )
    memory = Memory(config=memory_config)

    digest(args.query, args.user_id, memory=memory)
