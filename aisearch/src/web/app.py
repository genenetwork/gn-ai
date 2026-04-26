import os
import json
from itertools import groupby
import quart_flask_patch  # noqa: F401
import dspy
import torch
from quart import Quart, jsonify, request, render_template, Response
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gnais.config import Config
from gnais.search import HybridSearch
from markupsafe import escape

app = Quart(__name__)
app.config.from_object(Config)

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
torch.manual_seed(app.config["SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(app.config["SEED"])

llm = None
if app.config["MODEL_TYPE"]:
    llm = dspy.LM(
        app.config["MODEL_NAME"],
        api_key=app.config["API_KEY"],
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
else:
    llm = dspy.LM(
        model=f"openai/{app.config['MODEL_NAME']}",
        api_base="http://localhost:7501/v1",
        api_key="local",
        model_type="chat",
        max_tokens=10_000,
        n_ctx=10_000,
        seed=2_025,
        temperature=0,
        verbose=False,
    )

# KLUDGE: If you use a small local model, there is a very high
# likelihood that JSON validation will fail.
dspy.configure(lm=llm, adapter=dspy.JSONAdapter())

# KLUDGE: We create this here so that we don't have to keep
# re-instantiating this class on every request.
hybrid_search = HybridSearch()
hybrid_stream_search = HybridSearch(stream=True)


def _parse_output(raw_output: str):
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "Invalid JSON response from model",
            "raw_response": raw_output[:500],
        }


async def _run_search(query: str):
    """Run the hybrid search and return a parsed dict."""
    raw_output = await hybrid_search.handle(query)
    return _parse_output(raw_output)


def _group_results(results):
    """Group search results by type (defaulting to 'Other')."""
    sorted_results = sorted(results, key=lambda r: r.get("type") or "Other")
    grouped = {}
    for type_key, items in groupby(
        sorted_results, key=lambda r: r.get("type") or "Other"
    ):
        grouped[type_key] = list(items)
    return grouped


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


@app.route("/")
async def index():
    """Serve the AI search web interface."""
    return await render_template("index.html")


@app.route("/search/stream-shell", methods=["GET"])
@limiter.limit("300 per day")
async def search_stream_shell():
    """Shell endpoint that returns the streaming message container."""
    query = request.args.get("q", "").strip()
    if not query:
        return "<div class='error-message'>Missing query parameter 'q'</div>", 400
    if len(query) > 1000:
        return "<div class='error-message'>Query too long</div>", 400
    return await render_template("partials/stream_shell.html", query=query)


@app.route("/search/stream", methods=["GET"])
@limiter.limit("300 per day")
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

    async def event_stream():
        completed = set()
        final_sent = False

        yield _format_sse("search_state", _stream_status_markup("Streaming", "working"))
        yield _format_sse(
            "final_html",
            _stream_final_status_markup("Waiting for searches to complete…", "waiting"),
        )
        try:
            async for raw_event in hybrid_stream_search.handle(query):
                if final_sent:
                    break

                event = json.loads(raw_event)
                source = event.get("source")
                kind = event.get("kind")
                content = event.get("content", "")

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
                            _stream_status_markup("Complete", "complete"),
                        )
                        yield _format_sse(
                            "final_html",
                            _stream_final_status_markup(
                                "Synthesizing final answer…", "synthesizing"
                            ),
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
                            _stream_status_markup("Complete", "complete"),
                        )
                        yield _format_sse(
                            "final_html",
                            _stream_final_status_markup(
                                "Synthesizing final answer…", "synthesizing"
                            ),
                        )
                    continue

                if source == "hybrid" and kind == "final":
                    final_sent = True
                    parsed_output = _parse_output(content)
                    grouped = {}
                    if parsed_output.get("results"):
                        grouped = _group_results(parsed_output["results"])
                    template = app.jinja_env.get_template("partials/response_body.html")
                    final_html = await template.render_async(
                        query=query, data=parsed_output, grouped=grouped
                    )
                    yield _format_sse("final_html", final_html)
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
