from functools import lru_cache

import dspy


class Classification(dspy.Signature):
    input_text: str = dspy.InputField()
    decision: str = dspy.OutputField(desc='"keyword" or "semantic"')


class Extraction(dspy.Signature):
    input_text: str = dspy.InputField()
    keywords: str = dspy.OutputField()


@lru_cache(maxsize=2048)
def extract_keywords(query: str) -> str:
    """Extract list of keywords from query

    Args:
        query: user query

    Returns:
        list of keywords
    """
    return dspy.Predict(Extraction)(input_text=f"""
You are extremely good at extracting keywords from a search query related to specific entities (traits, markers, etc) in GeneNetwork.
Produce a list of space separated keywords featured in the query below. Only return that list.

{query}""")


@lru_cache(maxsize=2048)
def classify_search(query: str) -> str:
    """Classify user query as keyword search or semantic search

    Args:
        query: user query

    Returns:
        type of search for query processing
    """
    return dspy.Predict(Classification)(input_text=f"""
You are an experienced search classifier.
You can accurately tell from a query if a keyword search or semantic search is more appropriate to provide satisfactory answers to the user.
A keyword search is appropriate when specific entities feature in the query (i.e trait id, marker code, etc.).
A semantic search is better when the system needs to understand the meaning of the query and make implicit connections.
Infer the type of search that should be performed given the query below:

{query}
""")
