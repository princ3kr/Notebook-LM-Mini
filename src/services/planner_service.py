import networkx as nx
from typing import Dict, List, Tuple
from src.database.neo4j_conn import Neo4jConn
from fuzzywuzzy import fuzz, process

class PlannerService:
    def __init__(self, neo4j_conn: Neo4jConn, mastery: Dict[str, float], lam1=0.4, lam2=0.3, lam3=0.2, lam4=0.1):
        self.driver = neo4j_conn.connect()
        self.mastery = mastery
        self.lam1, self.lam2, self.lam3, self.lam4 = lam1, lam2, lam3, lam4
        self.G = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        if not self.driver: return G
        
        with self.driver.session() as session:
            nodes = session.run("""
                MATCH (c:Concept) 
                RETURN c.topic AS topic, 
                       c.difficulty_score AS difficulty, 
                       c.parent_unit AS parent_unit
            """)
            for record in nodes:
                G.add_node(
                    record["topic"],
                    difficulty=record["difficulty"] or 3.0,
                    parent_unit=record["parent_unit"]
                )
            
            edges = session.run("""
                MATCH (p:Concept)-[:PREREQUISITE_FOR]->(c:Concept)
                RETURN p.topic AS source, c.topic AS target
            """)
            for record in edges:
                G.add_edge(record["source"], record["target"])
        return G

    def _cost(self, u: str, v: str) -> float:
        mastery = self.mastery.get(v, 0.0)
        mastery_cost = 1.0 - mastery
        
        difficulty = self.G.nodes[v].get("difficulty", 3.0)
        difficulty_cost = (difficulty - 1) / 4.0
        
        in_degree_cost = self.G.in_degree(v) / max(len(self.G.nodes()), 1)
        out_degree_benefit = self.G.out_degree(v) / max(len(self.G.nodes()), 1)
        
        cost = (
            self.lam1 * mastery_cost +
            self.lam2 * difficulty_cost +
            self.lam3 * in_degree_cost -
            self.lam4 * out_degree_benefit
        )
        return max(cost, 0.001)

    def _get_sources_and_sinks(self, target_topics: List[str], threshold: float = 0.7) -> Tuple[List[str], List[str]]:
        sources = [node for node in self.G.nodes() if self.mastery.get(node, 0.0) >= threshold]
        if not sources:
            sources = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
            
        sinks = [
            topic
            for topic in target_topics
            if topic in self.G.nodes() and self.mastery.get(topic, 0.0) < threshold
        ]

        # Intent extraction may output a concept name that doesn't match `Concept.topic`
        # exactly. If no sinks are found, fuzzy-resolve each target to an existing node.
        if not sinks and target_topics:
            node_list = list(self.G.nodes())
            resolved_targets: List[str] = []
            for topic in target_topics:
                if topic in self.G.nodes():
                    resolved_targets.append(topic)
                    continue
                match = process.extractOne(topic, node_list, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= 60:
                    resolved_targets.append(match[0])

            sinks = [t for t in set(resolved_targets) if self.mastery.get(t, 0.0) < threshold]
        return sources, sinks

    def _compute_paths(self, sources: List[str], sinks: List[str]) -> Dict:
        paths = {}
        for src in sources:
            paths[src] = {}
            for sink in sinks:
                if src == sink:
                    # Trivial path: the learner can start with the sink concept directly.
                    # Without this, planned_paths can become empty for concepts that have
                    # no prerequisites (i.e., sink is also a valid starting source).
                    paths[src][sink] = {"path": [sink], "cost": 0.001}
                    continue
                try:
                    path = nx.dijkstra_path(self.G, src, sink, weight=lambda u, v, d: self._cost(u, v))
                    cost = sum(self._cost(path[i], path[i+1]) for i in range(len(path)-1))
                    paths[src][sink] = {"path": path, "cost": cost}
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        return paths

    def _greedy_set_cover(self, all_paths: Dict, sinks: List[str]) -> List[List[str]]:
        candidates = []
        for src, destinations in all_paths.items():
            for sink, data in destinations.items():
                candidates.append({
                    "path": data["path"],
                    "cost": data["cost"],
                    "covered": set(data["path"]) & set(sinks)
                })
        
        uncovered = set(sinks)
        selected = []
        while uncovered:
            best = max((c for c in candidates if c["covered"] & uncovered),
                       key=lambda c: (len(c["covered"] & uncovered), -c["cost"]), default=None)
            if not best: break
            selected.append(best["path"])
            uncovered -= best["covered"]
        return selected

    def plan(self, target_topics: List[str]) -> List[List[str]]:
        sources, sinks = self._get_sources_and_sinks(target_topics)
        all_paths = self._compute_paths(sources, sinks)

        any_path_found = any(all_paths[src] for src in all_paths)
        
        if not any_path_found and sinks:
            all_sources = list(self.G.nodes())
            all_paths = self._compute_paths(all_sources, sinks)

        return self._greedy_set_cover(all_paths, sinks)
