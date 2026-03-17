"""This module provides different constructs to interact with the LLM"""

__all__ = (
    "classify",
    "extract",
    "generate",
    "reformat",
    "stream_generate",
)

import dspy
import torch
from pydantic import BaseModel, Field


class Classification(dspy.Signature):
    """Classify search type"""
    input_text: str = dspy.InputField()
    decision: str = dspy.OutputField(desc='Either "keyword" or "semantic"')


classify = dspy.Predict(Classification)


class Extraction(dspy.Signature):
    """Extract keywords from query"""
    input_text: str = dspy.InputField()
    keywords: str = dspy.OutputField(desc="Space-separated keywords")


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


class StreamingGeneration(dspy.Signature):
    """Simple generation for streaming - must have string output"""

    input_text: str = dspy.InputField(desc="Query with instructions")
    chat_history: list = dspy.InputField(desc="Previous messages")
    context: list = dspy.InputField(desc="Retrieved documents")
    response: str = dspy.OutputField(desc="Answer to the query")


# For token-level streaming
stream_generate = dspy.Predict(StreamingGeneration)
