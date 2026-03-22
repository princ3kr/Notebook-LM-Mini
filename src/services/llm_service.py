import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.models.concept import Concept

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=1000,
            api_key=self.api_key
        )
        self.structured_llm = self.llm.with_structured_output(Concept, method="json_mode")

    def extract_concept(self, chunk_content: str, section: str = "", parent_unit: str = "Technical Document") -> Concept:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational knowledge graph builder specializing in electrical and electronics engineering.

Extract a single, specific, standalone technical concept from the given textbook chunk.

TOPIC:
- The single most specific technical idea in the chunk
- Must be a precise technical term, max 5 words
- Never use section headings or generic names like "Op-Amp Basics"
- Set to "SKIP" if chunk is introductory, transitional, or has no standalone concept

DESCRIPTION:
- Exactly 2 sentences, hard stop after 2nd sentence
- First sentence: what the concept is and how it works
- Second sentence: why it matters or what it enables
- Must be technical and precise, no filler

DIFFICULTY SCORE:
- REQUIRED. MUST BE A NUMBER (1.0 to 5.0).
- 1-2: definitional, no prerequisites
- 3: requires 1-2 prerequisites, moderate abstraction
- 4: multi-prerequisite, involves derivation or circuit analysis
- 5: heavy math, abstract reasoning, cross-topic synthesis

PREREQUISITES:
- MUST BE AN ARRAY OF STRINGS.
- Specific technical concept names only.
- Empty list [] if none.

EQUATIONS:
- REQUIRED. MUST BE AN ARRAY OF OBJECTS.
- Use an empty list [] if no equations are explicitly present.
- Only if explicitly present as a formula in the text.
- name: actual equation name, never a reference like "Equation 10.3".

SUBTOPICS:
- REQUIRED. MUST BE AN ARRAY OF STRINGS.
- Use an empty list [] if no subtopics are explicitly mentioned.
- Only specific sub-components explicitly mentioned in this chunk.

CHUNK TYPE:
- REQUIRED. Must be exactly one of: 'theory', 'derivation', 'numerical'
            
            Return ONLY a valid JSON object matching the requested schema."""),
            ("human", """Section: {section}
Parent Unit: {parent_unit}

Text:
{chunk_content}

CRITICAL RULES FOR JSON TYPES:
- parent_unit: MUST match the provided Parent Unit: {parent_unit}
- difficulty_score: MUST be a number, NOT a string (e.g., 3.0, not "3.0")
- prerequisites: MUST be an array [] OR [], NOT the string "null"
- equations: MUST be an array [] OR [], NOT the string "null" and NOT the JSON literal null
- subtopics: MUST be an array [] OR [], NOT the string "null" and NOT the JSON literal null
- Set topic to SKIP if no standalone technical concept exists
- chunk_type is REQUIRED""")
        ])
        
        chain = prompt | self.structured_llm
        return chain.invoke({
            "section": section,
            "parent_unit": parent_unit,
            "chunk_content": chunk_content
        })
