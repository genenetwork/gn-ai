def make_question_prompt(question: str) -> str:
    return f"Use all resources at your disposal to answer the following question: {question}"


question = make_question_prompt(
    question="Tell me traits and markers that have a lod score > 3 and an effect size > 1."
)
