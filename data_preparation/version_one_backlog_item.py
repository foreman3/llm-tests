from dataclasses import dataclass, field
from typing import Dict, Any, List
from data_preparation.documents import Document
import csv

@dataclass
class VersionOneBacklogItem(Document):
    title: str = ""
    description: str = ""
    acceptance_criteria: str = ""

    def __post_init__(self):
        # Populate the generic data dictionary with fields that might be used by steps.
        self.data["title"] = self.title
        self.data["description"] = self.description
        self.data["acceptance_criteria"] = self.acceptance_criteria

    @staticmethod
    def load_from_csv(filename: str) -> List['VersionOneBacklogItem']:
        documents = []
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                doc = VersionOneBacklogItem(
                    id=row['id'],
                    title=row['title'],
                    description=row['description'],
                    acceptance_criteria=row['acceptance_criteria']
                )
                documents.append(doc)
        return documents
