from pydantic import BaseModel, Field, field_validator
from typing import List, Any

class Question(BaseModel):
    concept: str
    question_text: str
    question_type: str
    expected_answer: str
    
class EvaluationResult(BaseModel):
    score: float
    feedback: str
    misconceptions: List[str] = Field(default_factory=list)

    @field_validator('misconceptions', mode='before')
    @classmethod
    def wrap_in_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        return v
