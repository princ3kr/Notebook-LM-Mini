from pydantic import BaseModel, Field
from typing import List, Optional

class Equation(BaseModel):
    name: str 
    latex: str 
    context: str 

class Concept(BaseModel):
    topic: str = Field(..., description="The unique name of the concept")
    chunk_type: str = Field(..., description="Type of chunk: 'theory', 'derivation', 'numerical'")
    description: str = Field(..., description="A concise 2-3 sentence technical explanation")
    equations: List[Equation] = Field(..., description="List of LaTeX formatted formulas related to this topic. Use an empty list [] if none.")
    subtopics: List[str] = Field(..., description="Specific smaller components or sub-chapters. Use an empty list [] if none.")
    prerequisites: List[str] = Field(..., description="List of topic names that must be understood first")
    difficulty_score: float = Field(..., description="Scale 1-5 of how complex the topic is")
    parent_unit: str = Field("Technical Document", description="The main Unit or Chapter this belongs to")
