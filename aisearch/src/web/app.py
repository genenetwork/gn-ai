import dspy
import torch

from gnais.config import Config
from gnais.rag import AISearch
from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


app = Flask(__name__)
app.config.from_object(Config)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per minute"],
    # KLUDGE: Consider moving this from RAM to Redis.
    storage_uri="memory://",
    strategy="fixed-window",
)

cache = Cache(config={"CACHE_TYPE": "SimpleCache"})
cache.init_app(app)

#  Bootstrapping our model
torch.manual_seed(app.config['SEED'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(app.config['SEED'])

llm = None
if app.config['MODEL_TYPE']:
    llm = dspy.LM(
        app.config['MODEL_NAME'],
        api_key=app.config['API_KEY'],
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
    corpus_path=app.config['CORPUS_PATH'],
    pcorpus_path=app.config['PCORPUS_PATH'],
    db_path=app.config['DB_PATH'],
)

# FIXME: There has to be a better way to do this
targeted_search = AISearch(
    corpus_path=app.config['CORPUS_PATH'],
    pcorpus_path=app.config['PCORPUS_PATH'],
    db_path=app.config['DB_PATH'],
    keyword_weight=0.7,
)


@app.route("/api/v1/search", methods=['GET'])
@limiter.limit("300 per day")
@cache.cached(timeout=604800)  # cache response for 1 week
def search():
    query = request.args.get('q')
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    if len(query) > 1000:  # limit query length
        return jsonify({"error": "Query too long"}), 400
    task_type = general_search.classify_search(query)
    if task_type.get("decision") == "keyword":
        output = targeted_search.handle(
            general_search.extract_keywords(query).get("keywords"))
        return output.model_dump_json(indent=4)
    output = general_search.handle(query)
    return output.model_dump_json(indent=4)
