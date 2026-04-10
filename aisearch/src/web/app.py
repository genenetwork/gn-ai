import quart_flask_patch  # noqa: F401
import dspy
import torch
from quart import Quart, jsonify, request
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gnais.config import Config
from gnais.ragent import HybridSearch

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


# KLUDGE: We create this here so that we don't have to keep
# re-instantiating this class on every request.
hybrid_search = HybridSearch()


@app.route("/api/v1/search", methods=["GET"])
@cache.cached(timeout=604800, make_cache_key=lambda: request.args.get("q"))  # cache response for 1 week
@limiter.limit("300 per day")
async def search(auth_token=None):
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    if len(query) > 1000:  # limit query length
        return jsonify({"error": "Query too long"}), 400

    # Log the authenticated user (optional - for audit purposes)
    user_id = get_user_id(auth_token)
    if user_id:
        app.logger.info("Search request from user: %s", user_id)

    output = await hybrid_search.handle(query)
    return output
