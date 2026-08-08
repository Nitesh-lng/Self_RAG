'''Here are the three Self-RAG grading points in English:

Document relevance (ISREL) — Is the retrieved document relevant? → yes/no
Hallucination (ISSUP) — Is the answer grounded in the documents? → yes/no
Answer usefulness (ISUSE) — Does the answer address the question? → yes/no'''


from pydantic import BaseModel,Field 

class GradeDocuments(BaseModel):
    """Binary score: is the retrieved document relevant to the question?"""
    binary_score: str = Field(description="Document relevant to question, 'yes' or 'no'")


class GradeHallucination(BaseModel):
    """Binary score: is the generated answer grounded in the documents?"""
    binary_score: str = Field(description="Answer grounded in documents, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    """Binary score: does the answer address the question?"""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
