import asyncio
import concurrent.futures
import os
import uuid
import quart_flask_patch  # noqa: F401
import dspy
import torch
import quart
import requests

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from quart import Quart, jsonify, request, render_template, Response, session, url_for, redirect, flash
from quart.typing import ResponseReturnValue
from quart_auth import AuthUser, current_user, login_required, login_user, logout_user, QuartAuth, Unauthorized
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gnais.config import Config
from gnais.search.ragent import hybrid_search
from web.settings import call_claude
from gnais.search.prompts import (
    GN_FACT_EXTRACTION_PROMPT,
    GN_UPDATE_MEMORY_PROMPT,
)
from markupsafe import escape

app = Quart(__name__)
app.config.from_object(Config)

# Basic Login
app.secret_key = Config.SECRET_KEY
QuartAuth(app)

# Set up template and static directories
template_dir = os.path.join(os.path.dirname(__file__), "templates")
app.template_folder = template_dir
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.static_folder = static_dir


limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per minute"],
    storage_uri="redis://localhost:6379",
    strategy="fixed-window",
)

cache = Cache(config={"CACHE_TYPE": "RedisCache"})
cache.init_app(app)

#  Bootstrapping our model
#  NOTE: The backend serving localhost:7501 should also run on the GPU
#  (e.g. vLLM with --device cuda) for maximum inference throughput.
torch.manual_seed(app.config["SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(app.config["SEED"])

LLM_CONFIG = {
    "model": app.config["MODEL_NAME"] if app.config.get("MODEL_NAME") else f"openai/{app.config['MODEL_NAME']}",
    "api_key": app.config["API_KEY"] if app.config.get("MODEL_NAME") else "local",
    "max_tokens": 20_000,
    "temperature": 0.5,
    "verbose": False
}

if not app.config.get("MODEL_TYPE"):
    LLM_CONFIG["api_base"] = "http://localhost:7501/v1"
    LLM_CONFIG["model_type"] = "chat"
    LLM_CONFIG["n_ctx"] = 100_000
    LLM_CONFIG["seed"] = 2_025


dspy.configure(lm=dspy.LM(**LLM_CONFIG))


@app.before_serving
async def _set_default_executor():
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=64))


#  Shared mem0 memory instance (loaded once at server startup)
_MEMORY = Memory(config=MemoryConfig(
    custom_fact_extraction_prompt=GN_FACT_EXTRACTION_PROMPT,
    custom_update_extraction_prompt=GN_UPDATE_MEMORY_PROMPT,
    llm={
        "provider": "litellm",
        "config": {
            "model": Config.MEMORY_MODEL,
            "temperature": 0.5,
            "max_tokens": 100_000,
            "api_key": Config.API_KEY,
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
            "host": "localhost",
            "port": 8001,
        },
    },
))

def _format_sse(event: str, data: str) -> str:
    lines = data.splitlines() or [""]
    payload = [f"event: {event}"]
    payload.extend(f"data: {line}" for line in lines)
    payload.append("")
    return "\n".join(payload) + "\n"


def _stream_chunk_markup(content: str, error: bool = False) -> str:
    css_class = "stream-fragment stream-fragment-error" if error else "stream-fragment"
    return f"<span class='{css_class}'>{escape(content)}</span>"


def _stream_status_markup(label: str, tone: str = "working") -> str:
    return f"<span class='stream-stage-badge is-{tone}'>{escape(label)}</span>"


def _stream_final_status_markup(message: str, tone: str = "waiting") -> str:
    icon = "🔍" if tone == "waiting" else "✨"
    return (
        f"<div class='stream-final-status is-{tone}'>"
        f"<div class='stream-final-status-icon'>{icon}</div>"
        f"<div class='stream-final-status-text'>{escape(message)}</div>"
        "</div>"
    )


@app.route("/login", methods=["GET", "POST"])
async def login():
    # KLUDGE: Set a static password for now to prevent token abuse
    # from non-GN testers.
    users = {"test@balg-qa.genenetwork.org": os.environ.get("USER_PASS")}
    if quart.request.method == "GET":
        return await render_template("login.html")
    form = await request.form
    email, password = form.get("email"), form.get("password")

    if email in users and users[email] == password:
        login_user(AuthUser(email))
        return redirect(url_for("settings"))

    await flash("Set the correct user/password")
    return redirect(url_for("login"))

@app.route('/logout')
async def logout():
    logout_user()
    return redirect(url_for("login"))

@app.errorhandler(Unauthorized)
async def redirect_to_login(*_: Exception) -> ResponseReturnValue:
    return redirect(url_for("login"))

@app.route("/settings", methods=["GET", "POST"])
@login_required
async def settings():
    """Let the user pick the LLM provider and set an API key."""
    providers = [
        ("genenetwork", "GeneNetwork"),
        ("claude-opus", "Claude Opus"),
        ("claude-haiku", "Claude Haiku"),
    ]

    if quart.request.method == "GET":
        selected = session.get("llm_provider", "genenetwork")
        return await render_template(
            "settings.html",
            providers=providers,
            selected=selected,
        )

    form = await request.form
    provider = form.get("provider", "genenetwork")
    api_key = form.get("api_key", "").strip()

    if provider != "genenetwork":
        try:
            call_claude(api_key)
        except requests.exceptions.RequestException as exc:
            await flash(f"Invalid API key or Anthropic request failed: {exc}")
            return redirect(url_for("settings"))

    session["llm_provider"] = provider
    if api_key:
        session["llm_api_key"] = api_key
    else:
        session.pop("llm_api_key", None)

    return redirect(url_for("index"))

@app.route("/")
@login_required
async def index():
    """Serve the AI search web interface."""
    return await render_template("index.html")


@app.route("/search/stream-shell", methods=["GET"])
@limiter.limit("300 per day")
@login_required
async def search_stream_shell():
    """Shell endpoint that returns the streaming message container."""
    query = request.args.get("q", "").strip()
    if not query:
        return "<div class='error-message'>Missing query parameter 'q'</div>", 400
    if len(query) > 1000:
        return "<div class='error-message'>Query too long</div>", 400
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return await render_template("partials/stream_shell.html", query=query)


@app.route("/search/stream", methods=["GET"])
@limiter.limit("300 per day")
@login_required
async def search_stream():
    """SSE endpoint for live hybrid search updates."""
    query = request.args.get("q", "").strip()
    if not query:
        return Response(
            _format_sse(
                "final_html",
                "<div class='error-message'>Missing query parameter 'q'</div>",
            ),
            mimetype="text/event-stream",
        )
    if len(query) > 1000:
        return Response(
            _format_sse(
                "final_html", "<div class='error-message'>Query too long</div>"
            ),
            mimetype="text/event-stream",
        )

    user_id = session.get("user_id")
    if not user_id:
        session["user_id"] = str(uuid.uuid4())
        user_id = session["user_id"]

    async def event_stream():
        completed = set()
        final_sent = False

        yield _format_sse("search_state", _stream_status_markup("Streaming", "working"))
        yield _format_sse(
            "final_html",
            _stream_final_status_markup("Waiting for searches to complete…", "waiting"),
        )
        try:
            async for event in hybrid_search(query, user_id=user_id, memory=_MEMORY):
                if final_sent:
                    break

                source = event["source"]
                kind = event["kind"]
                content = event["content"]

                if source in {"rag", "grag", "agent"} and kind == "status":
                    yield _format_sse(
                        f"{source}_chunk",
                        f"<div class='stream-status-msg'>{escape(content)}</div>",
                    )
                    continue

                if source in {"rag", "grag", "agent"} and kind == "chunk":
                    yield _format_sse(
                        f"{source}_chunk",
                        _stream_chunk_markup(content),
                    )
                    continue

                if source in {"rag", "grag", "agent"} and kind == "final":
                    yield _format_sse(
                        f"{source}_final_html",
                        str(content),
                    )
                    continue

                if source in {"rag", "grag", "agent"} and kind == "timing":
                    yield _format_sse(
                        f"{source}_timing",
                        escape(content),
                    )
                    continue

                if source in {"rag", "grag", "agent"} and kind == "error":
                    yield _format_sse(
                        f"{source}_chunk",
                        _stream_chunk_markup(content, error=True),
                    )
                    yield _format_sse(
                        f"{source}_done",
                        _stream_status_markup("Error", "error"),
                    )
                    completed.add(source)
                    if len(completed) == 3 and not final_sent:
                        yield _format_sse(
                            "search_state",
                            _stream_status_markup("Synthesizing", "working"),
                        )
                        yield _format_sse(
                            "final_html",
                            "<div class='synthesis-stream' sse-swap='synthesis_chunk' hx-swap='beforeend'></div>",
                        )
                    continue

                if source in {"rag", "grag", "agent"} and kind == "done":
                    yield _format_sse(
                        f"{source}_done",
                        _stream_status_markup("Complete", "complete"),
                    )
                    completed.add(source)
                    if len(completed) == 3 and not final_sent:
                        yield _format_sse(
                            "search_state",
                            _stream_status_markup("Synthesizing", "working"),
                        )
                        yield _format_sse(
                            "final_html",
                            "<div class='synthesis-stream' sse-swap='synthesis_chunk' hx-swap='beforeend'></div>",
                        )
                    continue

                if source == "synthesis" and kind == "chunk":
                    yield _format_sse("synthesis_chunk", content)
                    continue

                if source == "hybrid" and kind == "timing":
                    yield _format_sse(
                        "total_timing",
                        f"Total: {escape(content)}",
                    )
                    continue

                if source == "hybrid" and kind == "final":
                    final_sent = True
                    # await asyncio.to_thread(cache.set, cache_key, content, timeout=3600)
                    yield _format_sse("final_html", content)
                    yield _format_sse("search_state", _stream_status_markup("Complete", "complete"))
                    yield _format_sse("stream_end", "<div></div>")
                    return
        except Exception as exc:
            error_html = f"<div class='error-display'>{escape(str(exc))}</div>"
            yield _format_sse("final_html", error_html)
            yield _format_sse("stream_end", "<div></div>")

    response = Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    response.timeout = None
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
