from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
import math
import time

##############################################################################
# 1. Document & Data Type Definitions
##############################################################################

@dataclass
class Document:
    """
    A generic document for processing.
    Every document has a unique identifier, a dictionary of raw data,
    and a dictionary where processing steps can store their results.
    """
    id: str
    data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"Document(id={self.id}, data={self.data}, results={self.results})"

    def generate_prompt_text(self, fields: List[str] = None) -> str:
        """
        Generate the document text portion of the prompt.
        By default, include all fields in the format "Field Name": Field data.
        :param fields: Optional list of fields to include in the prompt.
        :return: A string representing the document text portion of the prompt.
        """
        if fields is None:
            fields = self.data.keys()
        return "\n".join([f"{field}: {self.data.get(field, '')}" for field in fields])
