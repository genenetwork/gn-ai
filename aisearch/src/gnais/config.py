"""This module provides different constructs to interact with the LLM"""
__all__ = (
    "classify",
    "extract",
    "generate",
)
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

    context: list = dspy.InputField(desc="Background information")
    input_text: str = dspy.InputField(desc="Query and instructions")
    feedback: ListInformation = dspy.OutputField(desc="System response to the query")


generate = dspy.Predict(Generation)
