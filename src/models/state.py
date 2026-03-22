from typing import TypedDict, List

class GraphState(TypedDict):
    student_id: str
    messages: List[dict]
    current_input: str
    target_topics: List[str]
    known_topics: List[str]
    current_concept: str
    current_question: str
    student_answer: str
    answer_score: float
    diagnosis_report: dict
    planned_paths: List[List[str]]
    current_path_index: int
    current_concept_index: int
    final_response: str
    is_transition: bool
    phase: str