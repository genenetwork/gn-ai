# This module formats the query to pass to the system


def make_query_prompt(query: str) -> str:
    return (
        f"Use all resources at your disposal to answer the following question: {query}"
    )

#question = input("We are ready for you. Please ask your question: ")
question = "Find all behavioral traits. Analyze their lod scores. Select only traits having a lod score greater or equal than 4 in the previous list of traits."

query = make_query_prompt(query=question)
