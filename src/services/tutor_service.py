from typing import List, Union
import json
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.models.state import GraphState
from src.models.diagnoses import Question, EvaluationResult
from src.services.intent_service import IntentService
from src.services.planner_service import PlannerService
from src.services.diagnoser_service import DiagnoserService
from src.database.student_db import StudentDB
from src.database.neo4j_conn import Neo4jConn

class TutorWorkflow:
    def __init__(self, neo4j_conn: Neo4jConn):
        self.neo4j_conn = neo4j_conn
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.3, 
            max_tokens=120
        )

    def _build_workflow(self):
        builder = StateGraph(GraphState)
        
        # Add all nodes
        builder.add_node("intent_parser", self.intent_parser_node)
        builder.add_node("planner", self.planner_node)
        builder.add_node("diagnoser_generate", self.diagnoser_generate_node)
        builder.add_node("human", self.human_node)
        builder.add_node("diagnoser_evaluate", self.diagnoser_evaluate_node)
        builder.add_node("tutor_respond", self.tutor_respond_node)
        
        # Define the starting point. Routing is handled via Command() in nodes.
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
        
        return Command(
            goto="diagnoser_generate",
            update={
                "planned_paths": planned_paths,
                "current_concept": planned_paths[0][0] if planned_paths else "",
                "current_path_index": 0,
                "current_concept_index": 0
            }
        )

    def diagnoser_generate_node(self, state: GraphState):
        diagnoser = DiagnoserService(self.neo4j_conn)
        q = diagnoser.generate_question(state["current_concept"])
        return Command(
            goto="human",
            update={
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
        return Command(
            goto="diagnoser_evaluate",
            update={"student_answer": answer}
        )

    def diagnoser_evaluate_node(self, state: GraphState):
        if state["student_answer"] == "__empty__":
            score, feedback, misconceptions = 0.0, "No answer provided.", ["Student did not attempt the question"]
        else:
            diagnoser = DiagnoserService(self.neo4j_conn)
            q = Question(**state["diagnosis_report"])
            evaluation = diagnoser.evaluate_answer(q, state["student_answer"])
            score, feedback, misconceptions = evaluation.score, evaluation.feedback, evaluation.misconceptions
            StudentDB(state["student_id"]).update_progress(state["current_concept"], score)
        
        attempts = state["diagnosis_report"].get("attempts", 0) + 1
        updates = {
            "answer_score": score,
            "diagnosis_report": {**state["diagnosis_report"], "feedback": feedback, "misconceptions": misconceptions, "attempts": attempts}
        }
        
        if score >= 0.7 or attempts >= 3:
            # Prepare to move to next concept
            updates["current_concept_index"] = state["current_concept_index"] + 1
            current_path = state["planned_paths"][state["current_path_index"]]
            if updates["current_concept_index"] < len(current_path):
                updates["current_concept"] = current_path[updates["current_concept_index"]]
            updates["current_question"] = ""
            updates["student_answer"] = ""
            updates["diagnosis_report"]["attempts"] = 0
            updates["is_transition"] = True
        else:
            updates["is_transition"] = False
            
        return Command(goto="tutor_respond", update=updates)

    def tutor_respond_node(self, state: GraphState):
        # We use the updated state in the response generation
        is_transition = state.get("is_transition", False)
        
        if is_transition:
            current_path = state["planned_paths"][state["current_path_index"]]
            if state["current_concept_index"] >= len(current_path):
                if state["current_path_index"] + 1 < len(state["planned_paths"]):
                    next_path_idx = state["current_path_index"] + 1
                    next_concept = state["planned_paths"][next_path_idx][0]
                    return Command(
                        goto="diagnoser_generate",
                        update={
                            "current_path_index": next_path_idx,
                            "current_concept_index": 0,
                            "current_concept": next_concept,
                            "final_response": f"✓ Good job! Moving to the next path starting with: **{next_concept}**"
                        }
                    )
                else:
                    return Command(
                        goto=END,
                        update={"final_response": "Congratulations! You have completed all planned learning paths."}
                    )
            else:
                return Command(
                    goto="diagnoser_generate",
                    update={"final_response": f"✓ Correct! Moving to next concept: **{state['current_concept']}**"}
                )
        else:
            # Address misconceptions
            misconceptions = state["diagnosis_report"].get("misconceptions", [])
            misconceptions_text = "\n".join([f"- {m}" for m in misconceptions])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert engineering tutor. 
                Give a SHORT explanation in maximum 3 sentences.
                Address only the specific misconceptions listed.
                No bullet points, no headers, just plain concise text.
                Stop after 3 sentences."""),
                ("human", """Concept: {concept}
                Misconceptions: {misconceptions}

                3 sentences only. Address misconceptions directly.""")
            ])
            
            explanation = (prompt | self.llm).invoke({
                "concept": state["current_concept"],
                "misconceptions": misconceptions_text
            })
            
            return Command(
                goto="diagnoser_generate",
                update={"final_response": f"Let me clarify **{state['current_concept']}**:\n\n{explanation.content}\n\nLet's try again:\n{state['current_question']}"}
            )
