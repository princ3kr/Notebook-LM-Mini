import json
import os
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process, fuzz
from src.models.concept import Concept
from src.database.neo4j_conn import Neo4jConn

class GraphService:
    def __init__(self, driver_conn: Neo4jConn):
        self.driver = driver_conn.connect()
        # Initialize embedding model lazily if needed
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_concepts_from_json(self, json_path: str) -> list[Concept]:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Exact reload logic
        concepts = []
        for item in data:
            concept = Concept(**item)
            concepts.append(concept)
        return concepts

    def build_graph(self, concepts: list[Concept]):
        if not self.driver: return

        with self.driver.session() as session:
            # Create nodes
            for concept in concepts:
                session.execute_write(self._create_concept_node, concept)
            
            # Create relationships
            existing_topics = [c.topic for c in concepts]
            for concept in concepts:
                # Part-Of first
                session.execute_write(self._create_part_of_relationship, concept)
                # Prerequisites
                for prereq in concept.prerequisites:
                    resolved = process.extractOne(prereq, existing_topics, scorer=fuzz.token_sort_ratio)
                    if resolved and resolved[1] >= 70 and resolved[0] != concept.topic:
                        session.execute_write(self._create_prerequisite_relationship, concept.topic, resolved[0])

    def build_graph_from_json(self, json_path: str):
        concepts = self.load_concepts_from_json(json_path)
        self.build_graph(concepts)

    def _create_concept_node(self, tx, concept: Concept):
        # Compute embeddings (Matching lines 499-500 of notebook)
        description_embedding = self.embedding_model.encode(concept.description).tolist()
        topic_embedding = self.embedding_model.encode(concept.topic).tolist()
        
        # Serialize equations (Matching lines 514-515 of notebook: str(list_of_dicts))
        equations_data = [eq.model_dump() for eq in concept.equations] if concept.equations else []
        
        query = """
        MERGE (c:Concept {topic: $topic})
        SET c.description = $description,
            c.difficulty_score = $difficulty_score,
            c.parent_unit = $parent_unit,
            c.chunk_type = $chunk_type,
            c.subtopics = $subtopics,
            c.equations = $equations,
            c.description_embedding = $description_embedding,
            c.topic_embedding = $topic_embedding
        """
        tx.run(query,
            topic=concept.topic,
            description=concept.description,
            difficulty_score=concept.difficulty_score,
            parent_unit=concept.parent_unit,
            chunk_type=concept.chunk_type,
            subtopics=concept.subtopics or [],
            equations=str(equations_data),
            description_embedding=description_embedding,
            topic_embedding=topic_embedding
        )

    def _create_prerequisite_relationship(self, tx, topic, prerequisite):
        query = """
        MATCH (c:Concept {topic: $topic})
        MATCH (p:Concept {topic: $prerequisite})
        MERGE (p)-[:PREREQUISITE_FOR]->(c)
        """
        tx.run(query, topic=topic, prerequisite=prerequisite)

    def _create_part_of_relationship(self, tx, concept: Concept):
        query = """
        MERGE (u:Unit {name: $parent_unit})
        MATCH (c:Concept {topic: $topic})
        MERGE (c)-[:PART_OF]->(u)
        """
        tx.run(query, parent_unit=concept.parent_unit, topic=concept.topic)
