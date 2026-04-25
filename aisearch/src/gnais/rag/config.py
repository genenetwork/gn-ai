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


class StreamGeneration(dspy.Signature):
    """Wrap generation interface"""

    input_text: str = dspy.InputField(desc="Query and instructions")
    chat_history: list = dspy.InputField(desc="History of conversation")
    context: list = dspy.InputField(desc="Background information")
    feedback: str = dspy.OutputField(
        desc="System response to the query that has a list of detailed answers and the final answer"
    )


generate = dspy.Predict(Generation)
generate_stream = dspy.Predict(StreamGeneration)


class Reformat(dspy.Signature):
    """Reformat ListInformation into valid Python dictionary"""

    input_text: str = dspy.InputField()
    result: str = dspy.OutputField(desc="Input reformatted to valid json format")


reformat = dspy.Predict(Reformat)


class SemanticReformulation(dspy.Signature):
    """Pick examples in the provided list that are semantically related to the user query and reformulate differently query using extracted terms for better search.
    Reformulated queries should not yet be a SPARQL query.
    Example: What is known about http://rdf.genenetwork.org/v1/id/phenotype_Skeletal_muscular__Grip_strength__mean_peak_force_of_3_trials_in_B6_BXD_F1_males_and_females_at_6_months_of_age_on_normal_chow_diet__all_Nash_Annex_cases___newtons_"""

    original_query: str = dspy.InputField(desc="User query")
    examples: list = dspy.InputField(desc="List of examples showing terms available in RDF")
    reformulated_queries: list[str] = dspy.OutputField(
        desc="List of reformulated queries from original terms using semantically close terms from the list of examples"
    )


reformulate = dspy.Predict(SemanticReformulation)


class SPARQLGenerator(dspy.Signature):
    """Generate a SPARQL SELECT query from a natural language question.
    Use the provided schema to construct valid queries. No syntax error will be accepted.
    Check the syntax and conformity to examples of the final queries before returning them.
    Exclude any query that does not meet the expectations"""

    original_query: str = dspy.InputField()
    rdf_classes: list = dspy.InputField(desc="RDF classes extracted from the graph")
    rdf_properties: list = dspy.InputField(
        desc="RDF properties extracted from the graph"
    )
    rdf_examples: list = dspy.InputField(
        desc="Real RDF examples in the graph that you can use to build correct SPARQL queries"
    )
    sparql_queries: list[str] = dspy.OutputField(
        desc="Exhaustive and valid SPARQL SELECT queries to retrieve relevant information and provide detailed answers to original query using RDF examples as baseline"
    )


generate_sparql = dspy.Predict(SPARQLGenerator)


class AnswerGenerator(dspy.Signature):
    """Generate a natural language answer from SPARQL query results and possible chat history."""

    requirements: str = dspy.InputField(desc="Set of instructions you must tightly follow")
    original_query: str = dspy.InputField(desc="Query provided")
    sparql_results: str = dspy.InputField(desc="JSON results from the SPARQL query")
    chat_history: list = dspy.InputField(desc="History of conversation")
    feedback: ListInformation = dspy.OutputField(desc="System response to the query")


class StreamAnswerGenerator(dspy.Signature):
    """Generate a streamed natural language answer from SPARQL query results."""

    requirements: str = dspy.InputField(desc="Set of instructions you must tightly follow")
    original_query: str = dspy.InputField(desc="Query provided")
    sparql_results: str = dspy.InputField(desc="JSON results from the SPARQL query")
    chat_history: list = dspy.InputField(desc="History of conversation")
    feedback: str = dspy.OutputField(
        desc="System response to the query that has a list of detailed answers and the final answer"
    )


generate_response = dspy.Predict(AnswerGenerator)
generate_response_stream = dspy.Predict(StreamAnswerGenerator)
