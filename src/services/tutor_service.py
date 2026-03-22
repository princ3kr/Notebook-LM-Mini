from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from src.database.neo4j_conn import Neo4jConn
from src.database.student_db import StudentDB
from src.models.diagnoses import Question
from src.models.state import GraphState
from src.services.diagnoser_service import DiagnoserService
from src.services.intent_service import IntentService
from src.services.planner_service import PlannerService
from src.services.tutor_lesson_utils import (
    PROCEED_PROMPT,
    format_equations_for_prompt,
    is_new_topic_intent,
    is_proceed_to_quiz,
)


class TutorWorkflow:
    def __init__(self, neo4j_conn: Neo4jConn):
        self.neo4j_conn = neo4j_conn
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.3, 
            max_tokens=250
        )
        # Longer explanations for the teach-first phase
        self.teach_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.35,
            max_tokens=1000,
        )

    def _build_workflow(self):
        builder = StateGraph(GraphState)
        
        # Add all nodes
        builder.add_node("intent_parser", self.intent_parser_node)
        builder.add_node("planner", self.planner_node)
        builder.add_node("tutor_teach", self.tutor_teach_node)
        builder.add_node("diagnoser_generate", self.diagnoser_generate_node)
        builder.add_node("human", self.human_node)
        builder.add_node("diagnoser_evaluate", self.diagnoser_evaluate_node)
        builder.add_node("tutor_respond", self.tutor_respond_node)
    
        builder.set_entry_point("intent_parser")
        
        return builder

    def intent_parser_node(self, state: GraphState):
        parser = IntentService(self.neo4j_conn)
        result = parser.parse(state["current_input"])
        return Command(
            goto="planner",
            update={
                "known_topics": result.known_topics,
                "target_topics": result.target_topics
            }
        )

    def planner_node(self, state: GraphState):
        db = StudentDB(state["student_id"])
        mastery = db.get_mastery()
        for t in state["known_topics"]:
            if t not in mastery: mastery[t] = 0.9
            
        planner = PlannerService(self.neo4j_conn, mastery)
        planned_paths = planner.plan(state["target_topics"])
        db.save_planned_path(planned_paths)

        # If there are no paths to learn (e.g., everything is already mastered),
        # end gracefully instead of letting later nodes index into an empty list.
        if not planned_paths:
            # More specific UX: distinguish "topic not recognized" vs "no path found".
            if not state["target_topics"]:
                msg = "I couldn't identify a learning topic from your message. Try: 'I want to learn about <concept name>'."
            else:
                msg = f"I couldn't find a learning path for: {', '.join(state['target_topics'])}. Try rephrasing or choosing a concept name from your knowledge base."
            return Command(
                goto=END,
                update={
                    "final_response": msg
                },
            )
        
        return Command(
            goto="tutor_teach",
            update={
                "planned_paths": planned_paths,
                "current_concept": planned_paths[0][0] if planned_paths else "",
                "current_path_index": 0,
                "current_concept_index": 0,
                "diagnosis_report": {},
                "current_question": "",
                "student_answer": "",
            }
        )

    def tutor_teach_node(self, state: GraphState):
        """Explain the current concept before any quiz (teach → then human confirms → quiz)."""
        concept_name = state.get("current_concept") or ""
        final_response = self._render_teach_lesson(concept_name)
        return Command(
            goto="human",
            update={
                "phase": "teach",
                "final_response": final_response,
                "current_question": PROCEED_PROMPT,
                "student_answer": "",
            },
        )

    def _render_teach_lesson(self, concept_name: str) -> str:
        """Build the lesson text (including equations if present)."""
        diagnoser = DiagnoserService(self.neo4j_conn)
        meta = diagnoser.fetch_concept_metadata(concept_name)
        equations_block, has_equations = format_equations_for_prompt(meta.get("equations"))

        equation_rules = ""
        if has_equations:
            equation_rules = """s
            - The knowledge base includes **equations** for this topic. You MUST include a section titled **### Key equations** (or similar).
            - For **each** equation listed below: state its **name**, write the **LaTeX** using markdown math (`$...$` for inline, `$$...$$` for display), and add **1–2 sentences** explaining what it represents (use the provided context; you may expand slightly for clarity).
            - Do not skip or paraphrase away the math — the student should see the actual formulas."""

        system_text = f"""You are an expert engineering tutor.
                Your job is to TEACH one concept before the student is quizzed.
                Rules:
                - Give a clear lesson: short overview, core definitions, key relationships, and one intuition or example if it helps.
                - Match depth to chunk_type (theory / derivation / numerical) when relevant.
                - Do NOT ask quiz questions or say "answer the following".
                - Use markdown sparingly (optional **bold** for key terms). Keep it readable in one scroll (~300–500 words max).
                - If metadata is thin, still teach from the concept name and general domain knowledge.{equation_rules}"""

        chunk_type = meta.get("chunk_type") or "theory"
        difficulty = meta.get("difficulty_score") or 3.0
        description = meta.get("description") or "(none)"
        subtopics = meta.get("subtopics") or []

        human_text = f"""Concept: {concept_name}
                Chunk type: {chunk_type}
                Difficulty (1–5): {difficulty}
                Description: {description}
            """
        if has_equations:
            human_text += f"""
                Equations from knowledge base (structured — use every one that appears here):
                {equations_block}
                """
        human_text += f"""
                Subtopics: {subtopics}

                Write the lesson."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                ("human", human_text),
            ]
        )

        lesson = (prompt | self.teach_llm).invoke({})
        body = lesson.content.strip()
        return f"## Lesson: **{concept_name}**\n\n{body}\n\n---\n{PROCEED_PROMPT}"

    def diagnoser_generate_node(self, state: GraphState):
        diagnoser = DiagnoserService(self.neo4j_conn)
        q = diagnoser.generate_question(state["current_concept"])
        return Command(
            goto="human",
            update={
                "phase": "quiz",
                "current_question": q.question_text,
                "diagnosis_report": {
                    "concept": q.concept,
                    "question_text": q.question_text,
                    "expected_answer": q.expected_answer,
                    "question_type": q.question_type,
                    "attempts": state["diagnosis_report"].get("attempts", 0)
                }
            }
        )

    def human_node(self, state: GraphState):
        student_answer = interrupt(state["current_question"])
        answer = student_answer.strip() if student_answer and student_answer.strip() else "__empty__"
        if state.get("phase") == "teach":
            return self._handle_teach_phase(state, answer)
        return self._handle_quiz_phase(state, answer)

    def _handle_teach_phase(self, state: GraphState, answer: str) -> Command:
        """Routing while waiting for user confirmation to start quiz."""
        if is_new_topic_intent(answer):
            return Command(
                goto="intent_parser",
                update={
                    "current_input": answer,
                    "student_answer": "",
                    "current_question": "",
                    "diagnosis_report": {},
                    "planned_paths": [],
                    "current_path_index": 0,
                    "current_concept_index": 0,
                    "answer_score": 0.0,
                    "is_transition": False,
                    "final_response": "",
                    "phase": "quiz",
                },
            )

        if is_proceed_to_quiz(answer):
            return Command(
                goto="diagnoser_generate",
                update={
                    "student_answer": "",
                    "final_response": "Starting the check-up for this concept. Answer the question below.",
                },
            )

        reminder = (
            "Say **I want to move further** or **I'm ready for the quiz** when you want to start the questions."
            if answer == "__empty__"
            else (
                "I didn't catch that. Say **I want to move further** or **ready for the quiz** to continue. "
                "Or start a new topic: **I want to learn about …**"
            )
        )
        return Command(goto="human", update={"final_response": reminder})

    def _handle_quiz_phase(self, state: GraphState, answer: str) -> Command:
        """Routing while waiting for recall-question answer."""
        if is_new_topic_intent(answer):
            return Command(
                goto="intent_parser",
                update={
                    "current_input": answer,
                    "student_answer": "",
                    "current_question": "",
                    "diagnosis_report": {},
                    "planned_paths": [],
                    "current_path_index": 0,
                    "current_concept_index": 0,
                    "answer_score": 0.0,
                    "is_transition": False,
                    "final_response": "",
                },
            )
        return Command(goto="diagnoser_evaluate", update={"student_answer": answer})

    def diagnoser_evaluate_node(self, state: GraphState):
        score, feedback, misconceptions = self._evaluate_or_default(state)

        attempts = state["diagnosis_report"].get("attempts", 0) + 1
        updates = {
            "answer_score": score,
            "diagnosis_report": {
                **state["diagnosis_report"],
                "feedback": feedback,
                "misconceptions": misconceptions,
                "attempts": attempts,
            },
        }

        return Command(goto="tutor_respond", update=self._updates_after_score(state, updates))

    def _evaluate_or_default(self, state: GraphState) -> tuple[float, str, list]:
        if state["student_answer"] == "__empty__":
            return (
                0.0,
                "No answer provided.",
                ["Student did not attempt the question"],
            )

        diagnoser = DiagnoserService(self.neo4j_conn)
        q = Question(**state["diagnosis_report"])
        evaluation = diagnoser.evaluate_answer(q, state["student_answer"])
        score, feedback, misconceptions = (
            evaluation.score,
            evaluation.feedback,
            evaluation.misconceptions,
        )
        StudentDB(state["student_id"]).update_progress(state["current_concept"], score)
        return score, feedback, misconceptions

    def _updates_after_score(self, state: GraphState, updates: dict) -> dict:
        """Add transition/retry fields based on score and attempt count."""
        attempts = updates["diagnosis_report"].get("attempts", 0)
        if not (updates["answer_score"] >= 0.7 or attempts >= 3):
            updates["is_transition"] = False
            return updates

        # Prepare to move to next concept
        updates["current_concept_index"] = state["current_concept_index"] + 1

        planned_paths = state.get("planned_paths", [])
        current_path_index = state.get("current_path_index", 0)
        if not planned_paths or current_path_index < 0 or current_path_index >= len(planned_paths):
            # Mark as transition so tutor_respond can safely end the session.
            updates["is_transition"] = True
            updates["final_response"] = (
                "Session learning path is unavailable or already completed. Ask for a new topic to continue."
            )
            return updates

        current_path = planned_paths[current_path_index]
        if updates["current_concept_index"] < len(current_path):
            updates["current_concept"] = current_path[updates["current_concept_index"]]

        updates["current_question"] = ""
        updates["student_answer"] = ""
        updates["diagnosis_report"]["attempts"] = 0
        updates["is_transition"] = True
        return updates

    def tutor_respond_node(self, state: GraphState):
        """After evaluation: either transition to the next lesson or address misconceptions."""
        if state.get("is_transition", False):
            return self._handle_transition(state)
        return self._handle_misconceptions(state)

    def _handle_transition(self, state: GraphState) -> Command:
        planned_paths = state.get("planned_paths", [])
        current_path_index = state.get("current_path_index", 0)
        if not planned_paths or current_path_index < 0 or current_path_index >= len(planned_paths):
            return Command(
                goto=END,
                update={
                    "final_response": state.get(
                        "final_response",
                        "You have completed the planned learning paths.",
                    )
                },
            )

        current_path = planned_paths[current_path_index]
        if state["current_concept_index"] >= len(current_path):
            if current_path_index + 1 < len(planned_paths):
                next_path_idx = current_path_index + 1
                next_concept = planned_paths[next_path_idx][0]
                return Command(
                    goto="tutor_teach",
                    update={
                        "current_path_index": next_path_idx,
                        "current_concept_index": 0,
                        "current_concept": next_concept,
                        "diagnosis_report": {},
                        "current_question": "",
                        "student_answer": "",
                        "phase": "teach",
                        "final_response": f"✓ Good job! Next path — here's the lesson for **{next_concept}**.",
                    },
                )

            last_concept = current_path[-1] if current_path else state.get("current_concept", "this topic")
            return Command(
                goto=END,
                update={
                    "final_response": (
                        f"Nice work — you've finished **{last_concept}** in this session "
                        f"({len(planned_paths)} learning path(s) total). "
                        "Ask **I want to learn about …** anytime to start another topic."
                    )
                },
            )

        return Command(
            goto="tutor_teach",
            update={
                "diagnosis_report": {},
                "current_question": "",
                "student_answer": "",
                "phase": "teach",
                "final_response": f"✓ Nice work! Next concept: **{state['current_concept']}** — lesson below.",
            },
        )

    def _handle_misconceptions(self, state: GraphState) -> Command:
        misconceptions = state["diagnosis_report"].get("misconceptions", [])
        misconceptions_text = "\n".join([f"- {m}" for m in misconceptions])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert engineering tutor.
                Give a SHORT explanation in maximum 3 sentences.
                Address only the specific misconceptions listed.
                No bullet points, no headers, just plain concise text.
                Stop after 3 sentences.""",
                ),
                (
                    "human",
                    """Concept: {concept}
                Misconceptions: {misconceptions}

                3 sentences only. Address misconceptions directly.""",
                ),
            ]
        )

        explanation = (prompt | self.llm).invoke(
            {"concept": state["current_concept"], "misconceptions": misconceptions_text}
        )

        return Command(
            goto="diagnoser_generate",
            update={
                "final_response": (
                    f"Let me clarify **{state['current_concept']}**:\n\n{explanation.content}\n\n"
                    f"Let's try again:\n{state['current_question']}"
                )
            },
        )
