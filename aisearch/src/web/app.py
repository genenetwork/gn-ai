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
from gnais.ragent import HybridSearch

app = Quart(__name__)
app.config.from_object(Config)

# Set up template and static directories
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app.template_folder = template_dir
static_dir = os.path.join(os.path.dirname(__file__), 'static')
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


async def _run_search(query: str):
    """Run the hybrid search and return a parsed dict."""
    raw_output = await hybrid_search.handle(query)
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "Invalid JSON response from model",
            "raw_response": raw_output[:500],
        }


def _group_results(results):
    """Group search results by type (defaulting to 'Other')."""
    sorted_results = sorted(results, key=lambda r: r.get("type") or "Other")
    grouped = {}
    for type_key, items in groupby(sorted_results, key=lambda r: r.get("type") or "Other"):
        grouped[type_key] = list(items)
    return grouped


@app.route("/")
async def index():
    """Serve the AI search web interface."""
    return await render_template("index.html")


@app.route("/api/v1/search", methods=["GET"])
@cache.cached(timeout=604800, make_cache_key=lambda: f"api:v1:{request.args.get('q')}")
@limiter.limit("300 per day")
async def search():
    """JSON search endpoint - no authentication required."""
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    if len(query) > 1000:
        return jsonify({"error": "Query too long"}), 400

    parsed_output = await _run_search(query)
    if parsed_output.get("status") == "error" or "error" in parsed_output:
        status_code = 500 if parsed_output.get("status") == "error" else 200
        return jsonify(parsed_output), status_code
    return jsonify(parsed_output)


@app.route("/search", methods=["GET"])
@cache.cached(timeout=604800, make_cache_key=lambda: f"html:v1:{request.args.get('q')}")
@limiter.limit("300 per day")
async def search_html():
    """HTML fragment search endpoint for htmx."""
    query = request.args.get("q")
    if not query:
        return "<div class='error-message'>Missing query parameter 'q'</div>", 400
    if len(query) > 1000:
        return "<div class='error-message'>Query too long</div>", 400

    parsed_output = await _run_search(query)
    grouped = {}
    if parsed_output.get("results"):
        grouped = _group_results(parsed_output["results"])
    return await render_template("partials/response.html", query=query, data=parsed_output, grouped=grouped)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
