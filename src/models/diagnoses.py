from pydantic import BaseModel
from typing import List

class Question(BaseModel):
    concept: str
    question_text: str
    question_type: str
    expected_answer: str
    
class EvaluationResult(BaseModel):
    score: float
    feedback: str
    misconceptions: List[str]
