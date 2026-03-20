from pydantic import BaseModel
from typing import List

class IntentOutput(BaseModel):
    known_topics: List[str]
    target_topics: List[str]
