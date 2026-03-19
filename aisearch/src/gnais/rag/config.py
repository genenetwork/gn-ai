"""This module provides different constructs to interact with the LLM"""

import dspy
import torch
from pydantic import BaseModel, Field


class Classification(dspy.Signature):
    input_text: str = dspy.InputField()
    decision: str = dspy.OutputField(desc='"keyword" or "semantic"')


classify = dspy.Predict(Classification)


class Extraction(dspy.Signature):
    input_text: str = dspy.InputField()
    keywords: str = dspy.OutputField()


extract = dspy.Predict(Extraction)


class Information(BaseModel):
    """Extract relevant information for query"""

    answer: str = Field(
        description="Specific point addressing the query from the context"
    )
    links: list[str] = Field(
        description="All links associated to RDF entities related to the point"
    )


class ListInformation(BaseModel):
    """Address recursively a query"""

    detailed_answers: list[Information] = Field(
        description="List of answers to the query"
    )
    final_answer: str = Field(
        description="Synthesized and comprehensive answer using detailed answers"
    )


class Generation(dspy.Signature):
    """Wrap generation interface"""

    input_text: str = dspy.InputField(desc="Query and instructions")
    chat_history: list = dspy.InputField(desc="History of conversation")
    context: list = dspy.InputField(desc="Background information")
    feedback: ListInformation = dspy.OutputField(desc="System response to the query")


generate = dspy.Predict(Generation)


class Reformat(dspy.Signature):
    """Reformat ListInformation into valid Python dictionary"""

    input_text: str = dspy.InputField()
    result: str = dspy.OutputField(desc="Input reformatted to valid Python dictionary")


reformat = dspy.Predict(Reformat)


class SPARQLGenerator(dspy.Signature):
    """Generate a SPARQL SELECT query from a natural language question.
    Use the provided schema to construct valid queries."""

    original_query: str = dspy.InputField(desc="Query provided")
    classes_info: str = dspy.InputField(desc="Mapping for available classes")
    properties_info: str = dspy.InputField(desc="Mapping for available properties")
    sparql_queries: list[str] = dspy.OutputField(
        desc="Max 50 valid SPARQL SELECT queries that can retrieve any relevant information. Do not assume perfect match of entity (subject, object). Queries should use class and property information to try different variations of the keywords in the query and find successful queries."
    )


generate_sparql = dspy.Predict(SPARQLGenerator)


class AnswerGenerator(dspy.Signature):
    """Generate a natural language answer from SPARQL query results."""

    original_query: str = dspy.InputField(desc="Query provided")
    sparql_results: str = dspy.InputField(desc="JSON results from the SPARQL query")
    chat_history: list = dspy.InputField(desc="History of conversation")
    feedback: str = dspy.OutputField(desc="System response to the query")


generate_response = dspy.Predict(AnswerGenerator)
