import quart_flask_patch  # noqa: F401
import dspy
import torch
import json
import asyncio

from quart import Quart, jsonify, request, abort, make_response
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gnais.config import Config
from gnais.rag import AISearch, classify_search, extract_keywords

app = Quart(__name__)
app.config.from_object(Config)

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
        # FIXME: Make this configurable
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


# KLUDGE: We create these here so that we don't have to keep
# re-instantiating this class on every request.
general_search = AISearch(
    corpus_path=app.config["CORPUS_PATH"],
    pcorpus_path=app.config["PCORPUS_PATH"],
    db_path=app.config["DB_PATH"],
)

# FIXME: There has to be a better way to do this
targeted_search = AISearch(
    corpus_path=app.config["CORPUS_PATH"],
    pcorpus_path=app.config["PCORPUS_PATH"],
    db_path=app.config["DB_PATH"],
    keyword_weight=0.7,
)


class ServerSentEvent:
    """Helper class for formatting Server-Sent Events (SSE)."""

    def __init__(self, data: str, event: str = None):
        self.data = data
        self.event = event

    def encode(self) -> str:
        """Encode the event as a properly formatted SSE message."""
        lines = []
        if self.event:
            lines.append(f"event: {self.event}")
        for line in self.data.split('\n'):
            lines.append(f"data: {line}")
        lines.append('')
        return '\n'.join(lines) + '\n'


def format_sse_event(data: dict, event: str = None) -> str:
    """Format a dictionary as an SSE event string."""
    return ServerSentEvent(data=json.dumps(data), event=event).encode()


@app.route("/api/v1/search/stream", methods=["GET"])
@limiter.limit("100 per day")
async def search_stream():
    """Stream search results with intermediate status updates via SSE."""
    if "text/event-stream" not in request.accept_mimetypes:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    if len(query) > 500:
        return jsonify({"error": "Query too long"}), 400

    async def generate_stream():
        """Async generator that yields SSE events with status updates and token streaming."""
        loop = asyncio.get_event_loop()

        # Status: Start
        yield format_sse_event({"type": "status", "message": "Starting search...", "step": "start"}, event="status")

        # Step 1: Classify search type
        yield format_sse_event({"type": "status", "message": "Classifying search type...", "step": "classify"}, event="status")
        task_type = await loop.run_in_executor(None, classify_search, query)
        is_keyword = task_type.get("decision") == "keyword"
        yield format_sse_event({"type": "status", "message": f"Search classified as: {'keyword' if is_keyword else 'semantic'}", "step": "classified"}, event="status")

        # Step 2: Extract keywords if needed
        if is_keyword:
            yield format_sse_event({"type": "status", "message": "Extracting keywords...", "step": "extract"}, event="status")
            keywords_result = await loop.run_in_executor(None, extract_keywords, query)
            processed_query = keywords_result.get("keywords", query)
            yield format_sse_event({"type": "status", "message": f"Keywords: {processed_query}", "step": "extracted"}, event="status")
        else:
            processed_query = query

        # Step 3: Retrieve documents
        yield format_sse_event({"type": "status", "message": "Retrieving relevant documents...", "step": "retrieve"}, event="status")
        search_instance = targeted_search if is_keyword else general_search

        # Step 4: Generate response with token streaming
        yield format_sse_event({"type": "status", "message": "Generating response...", "step": "generate"}, event="status")

        try:
            # Use token streaming
            token_index = 0
            full_response = []

            async for event_type, data in search_instance.ahandle_token_stream(processed_query):
                if event_type == "token":
                    yield format_sse_event({"type": "token", "content": data, "index": token_index}, event="token")
                    full_response.append(data)
                    token_index += 1
                    await asyncio.sleep(0)
                elif event_type == "prediction":
                    # Final prediction received
                    pass
                elif event_type == "error":
                    yield format_sse_event({"type": "error", "message": data}, event="error")
                    return

            yield format_sse_event({"type": "status", "message": "Response generated", "step": "complete"}, event="status")
            yield format_sse_event({"type": "result", "data": "".join(full_response)}, event="result")
        except Exception as e:
            yield format_sse_event({"type": "error", "message": str(e)}, event="error")

    response = await make_response(
        generate_stream(),
        {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'X-Accel-Buffering': 'no',
        },
    )
    response.timeout = None
    return response
