import os
from datetime import datetime
from typing import Dict, List, Optional
from tinydb import TinyDB, Query

class StudentDB:
    def __init__(self, student_id: str, db_folder: str = "sessions"):
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
        
        self.db_path = os.path.join(db_folder, "student_sessions.json")
        self.student_id = student_id
        self.db = TinyDB(self.db_path)
        self.Student = Query()
        self._ensure_student_exists()

    def _ensure_student_exists(self):
        result = self.db.search(self.Student.student_id == self.student_id)
        if not result:
            self.db.insert({
                "student_id": self.student_id,
                "mastery": {},
                "planned_path": [],
                "progress": {},
                "last_json_path": None,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            })

    def get_user_data(self) -> dict:
        result = self.db.search(self.Student.student_id == self.student_id)
        return result[0] if result else {}

    def update_last_json_path(self, json_path: str):
        self.db.update(
            {"last_json_path": json_path, "last_updated": datetime.now().isoformat()},
            self.Student.student_id == self.student_id
        )

    def get_last_json_path(self) -> Optional[str]:
        data = self.get_user_data()
        return data.get("last_json_path")

    def save_planned_path(self, paths: List[List[str]]):
        self.db.update(
            {"planned_path": paths, "last_updated": datetime.now().isoformat()},
            self.Student.student_id == self.student_id
        )

    def get_planned_path(self) -> List[List[str]]:
        return self.get_user_data().get("planned_path", [])

    def update_progress(self, topic: str, score: float):
        data = self.get_user_data()
        progress = data.get("progress", {})
        
        if topic not in progress:
            progress[topic] = {"attempts": 0, "cumulative_score": 0.0}
        
        progress[topic]["attempts"] += 1
        progress[topic]["cumulative_score"] += score
        
        new_mastery = progress[topic]["cumulative_score"] / progress[topic]["attempts"]
        
        self.db.update(
            {"progress": progress, "last_updated": datetime.now().isoformat()},
            self.Student.student_id == self.student_id
        )
        self.update_mastery(topic, new_mastery)

    def update_mastery(self, topic: str, score: float):
        data = self.get_user_data()
        mastery = data.get("mastery", {})
        mastery[topic] = round(max(0.0, min(1.0, score)), 2)
        self.db.update(
            {"mastery": mastery, "last_updated": datetime.now().isoformat()},
            self.Student.student_id == self.student_id
        )

    def get_mastery(self) -> Dict[str, float]:
        return self.get_user_data().get("mastery", {})

    def get_next_concept(self) -> Optional[str]:
        planned_path = self.get_planned_path()
        mastery = self.get_mastery()
        for path in planned_path:
            for concept in path:
                if mastery.get(concept, 0.0) < 0.7:
                    return concept
        return None
