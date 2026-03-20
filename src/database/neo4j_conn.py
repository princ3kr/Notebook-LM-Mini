import os
from neo4j import GraphDatabase

class Neo4jConn:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None

    def connect(self):
        if not self.uri or not self.user or not self.password:
            return None
        if not self.driver:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            self.driver.verify_connectivity()
            return self.driver
        except Exception as e:
            print(f"Neo4j connectivity failed: {e}")
            return None

    def is_connected(self) -> bool:
        driver = self.connect()
        return driver is not None

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
