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
    data: dict[str, float] = Field(description="Measurements or statistics fetched from the context. Do not hallucinate data. If no data is available, say NA")
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
    data_table: str = Field(
        description="Data in detailed answers regrouped and properly formatted as a table for comparison by user"
    )


class Generation(dspy.Signature):
    """Wrap generation interface"""

    input_text: str = dspy.InputField(desc="Query and instructions")
    chat_history: list = dspy.InputField(desc="History of conversation")
    context: list = dspy.InputField(desc="Background information")
    feedback: ListInformation = dspy.OutputField(desc="System response to the query")


generate = dspy.Predict(Generation)


class Reformat(dspy.Signature):
    """Reformat ListInformation into valid Markdown Key-Value format using header-based hierarchy"""

    input_text: str = dspy.InputField()
    result: str = dspy.OutputField(
        desc="""
        List of informations formatted as Markdown Key-Value using header-based hierarchy defined as below:
        # Answers
        ## Answer 1
        - **Text:** {text}
        - **Associated data:** {data}
        - **Trait link 1:** {link}
        - **Trait link 2:** {link}

        ## Answer 2
        - **Text:** {text}
        - **Associated data:** {data}
        - **Trait link 1:** {link}
        - **Trait link 2:** {link}

        ## Final answer
        - **Text:** {text}

        ## Data table
        |{variable 1}|{variable 2}|
        ---------------------------
        |{value 1}|{value 1}|
        |{value 2}|{value 2}|

        Do not include links in data table.
        """
    )


reformat = dspy.Predict(Reformat)


class SPARQLGenerator(dspy.Signature):
    """Generate a SPARQL SELECT query from a natural language question.
    Use the provided schema to construct valid queries."""

    original_query: str = dspy.InputField(desc="User query")
    classes_info: str = dspy.InputField(desc="Mapping for available classes")
    properties_info: str = dspy.InputField(desc="Mapping for available properties")
    sparql_queries: list[str] = dspy.OutputField(
        desc="As many and exhaustive SPARQL SELECT queries that you can generate and that can retrieve all relevant information necessary to provide detailed answer to the user query."
    )


generate_sparql = dspy.Predict(SPARQLGenerator)


class AnswerGenerator(dspy.Signature):
    """Generate a natural language answer from SPARQL query results."""

    original_query: str = dspy.InputField(desc="Query provided")
    sparql_results: str = dspy.InputField(desc="JSON results from the SPARQL query")
    chat_history: list = dspy.InputField(desc="History of conversation")
    feedback: ListInformation = dspy.OutputField(desc="System response to the query")


generate_response = dspy.Predict(AnswerGenerator)
