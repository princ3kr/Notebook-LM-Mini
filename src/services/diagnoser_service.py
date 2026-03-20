from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.models.diagnoses import Question, EvaluationResult
from src.database.neo4j_conn import Neo4jConn

class DiagnoserService:
    def __init__(self, neo4j_conn: Neo4jConn):
        self.driver = neo4j_conn.connect()
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=300)

    def _fetch_concept(self, topic: str) -> dict:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {topic: $topic})
                RETURN c.description AS description,
                       c.equations AS equations,
                       c.difficulty_score AS difficulty_score,
                       c.chunk_type AS chunk_type,
                       c.subtopics AS subtopics
            """, topic=topic)
            record = result.single()
            return dict(record) if record else {}

    def generate_question(self, topic: str) -> Question:
        concept = self._fetch_concept(topic)
        structured_llm = self.llm.with_structured_output(Question)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert engineering tutor generating assessment questions.

            Generate ONE question for the given concept based on its type:
            - theory → conceptual question testing understanding of how/why
            - derivation → equation-based question requiring mathematical reasoning  
            - numerical → problem-solving question with specific values

            Rules:
            - question must be specific to the concept, not generic
            - expected_answer must be a complete reference answer
            - question_type must match chunk_type"""),
            ("human", """Concept: {topic}
            Description: {description}
            Chunk Type: {chunk_type}
            Difficulty: {difficulty_score}
            Equations: {equations}
            Subtopics: {subtopics}

            Generate a question that tests deep understanding of this concept.""")
        ])
        
        return (prompt | structured_llm).invoke({
            "topic": topic,
            "description": concept.get("description", ""),
            "chunk_type": concept.get("chunk_type", "theory"),
            "difficulty_score": concept.get("difficulty_score", 3.0),
            "equations": concept.get("equations", ""),
            "subtopics": concept.get("subtopics", [])
        })

    def evaluate_answer(self, question: Question, student_answer: str) -> EvaluationResult:
        structured_llm = self.llm.with_structured_output(EvaluationResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert engineering tutor evaluating a student's answer.

            Score the answer from 0.0 to 1.0:
            - 1.0: complete, accurate, shows deep understanding
            - 0.7-0.9: mostly correct with minor gaps
            - 0.4-0.6: partial understanding, significant gaps
            - 0.1-0.3: minimal understanding, major misconceptions
            - 0.0: completely wrong or no attempt

            Rules:
            - feedback must be specific and constructive
            - misconceptions must identify exact conceptual gaps
            - be strict but fair"""),
            ("human", """Concept: {concept}
            Question: {question_text}
            Expected Answer: {expected_answer}
            Student Answer: {student_answer}

            Evaluate the student's answer.""")
        ])
        
        return (prompt | structured_llm).invoke({
            "concept": question.concept,
            "question_text": question.question_text,
            "expected_answer": question.expected_answer,
            "student_answer": student_answer
        })
