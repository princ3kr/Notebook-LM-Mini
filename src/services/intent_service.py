import os
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.models.intent import IntentOutput

class IntentService:
    def __init__(self, neo4j_conn):
        self.driver = neo4j_conn.connect()
        self.biencoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.crossencoder = CrossEncoder('cross-encoder/stsb-roberta-base')
        self.topics, self.topic_embeddings = self._load_node_embeddings()

    def _load_node_embeddings(self):
        if not self.driver:
            return [], np.array([])
            
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                RETURN c.topic AS topic, c.topic_embedding AS embedding
            """)
            topics, embeddings = [], []
            for record in result:
                topics.append(record["topic"])
                embeddings.append(record["embedding"])
        
        if not embeddings:
            return topics, np.array([])
        return topics, np.array(embeddings)

    def _extract_intent(self, student_prompt: str) -> IntentOutput:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        structured_llm = llm.with_structured_output(IntentOutput)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent extractor for a learning system.

Extract TWO things from the student message:
1. known_topics: ONLY concepts the student EXPLICITLY says they know
2. target_topics: ONLY concepts the student EXPLICITLY says they want to learn

Rules:
- Never extract vague terms like "basic amplifiers" or "circuits"
- Always extract the most specific technical concept name possible
- known_topics must be EXPLICITLY stated as known — never infer
- target_topics must be EXPLICITLY stated as goals — never infer
- "from scratch" or "everything" → return empty lists for both
- "I don't understand X" → target: [X], known: []
- Maximum 3 topics per list

Examples:
- "I know basic amplifiers" → known: ["differential amplifier", "inverting amplifier"], target: []
- "I want to learn filters" → known: [], target: ["low pass filter"]
- "I don't understand virtual ground" → known: [], target: ["virtual ground"]
- "from scratch" → known: [], target: []

Return empty lists only if truly not mentioned."""),
            ("human", "{student_prompt}")
        ])
        
        return (prompt | structured_llm).invoke({"student_prompt": student_prompt})

    def _retrieve_candidates(self, query: str, top_k: int = 10) -> List[str]:
        query_embedding = self.biencoder.encode([query])
        similarities = cosine_similarity(query_embedding, self.topic_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.topics[i] for i in top_k_indices]

    def _rerank_candidates(self, query: str, candidates: List[str], threshold: float = 0.0) -> Optional[str]:
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.crossencoder.predict(pairs)
        best_idx = np.argmax(scores)
        if scores[best_idx] < threshold:
            return None
        return candidates[best_idx]

    def parse(self, student_prompt: str) -> IntentOutput:
        raw_intent = self._extract_intent(student_prompt)
        
        resolved_known = []
        for topic in raw_intent.known_topics:
            candidates = self._retrieve_candidates(topic)
            resolved = self._rerank_candidates(topic, candidates)
            if resolved: resolved_known.append(resolved)
            
        resolved_targets = []
        for topic in raw_intent.target_topics:
            candidates = self._retrieve_candidates(topic)
            resolved = self._rerank_candidates(topic, candidates)
            if resolved: resolved_targets.append(resolved)
            
        return IntentOutput(known_topics=resolved_known, target_topics=resolved_targets)
