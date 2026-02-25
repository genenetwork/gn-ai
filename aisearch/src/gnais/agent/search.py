"""This is the main module of the package using agent tool calling"""

import torch

from gnais.agent.agent import *

QUERY = os.getenv("QUERY")
if QUERY is None:
    raise ValueError("QUERY must be specified for program to run")
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


dspy.configure(lm=llm, adapter=dspy.JSONAdapter())


def search(query: str):
    system_prompt = """
            You excel at addressing search query using the context you have. You do not make mistakes.
            Extract answers to the query from the context and provide links associated with each RDF entity.
            All links pointing to specific traits should be translated to CD links using the trait id and the dataset name.
            Original trait link: https://rdf.genenetwork.org/v1/id/trait_BXDPublish_16339
            Trait id: 16339
            Dataset name: BXDPublish
            New trait link: https://cd.genenetwork.org/show_trait?trait_id=16339&dataset=BXDPublish
            \n
            """
    final_query = system_prompt + query
    search = AISearch()
    output = search(query=final_query).get("solution")
    return output.model_dump_json(indent=4)
