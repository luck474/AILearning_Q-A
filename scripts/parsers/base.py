"""
Base parser interface for question bank files.
"""
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseParser(ABC):
    """Abstract base class for question parsers."""
    
    # Question type mappings
    TYPE_TRUE_FALSE = "true_false"
    TYPE_MULTIPLE_CHOICE = "multiple_choice"
    TYPE_MULTIPLE_RESPONSE = "multiple_response"
    TYPE_FILL_BLANK = "fill_in_the_blank"
    TYPE_UNKNOWN = "unknown"
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.questions: List[Dict[str, Any]] = []
    
    @abstractmethod
    def parse(self) -> List[Dict[str, Any]]:
        """Parse the source file and return a list of question dicts."""
        pass
    
    def save_json(self, output_path: str) -> None:
        """Save parsed questions to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.questions, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.questions)} questions to {output_path}")
    
    @staticmethod
    def extract_option_key(option_text: str) -> str:
        """Extract the letter key (A, B, C, D) from an option string."""
        match = re.match(r'^([A-Z])[\\.、]', option_text)
        return match.group(1) if match else ""
    
    @staticmethod
    def normalize_answer(raw_answer: str) -> str:
        """Extract answer letters from raw answer text."""
        # Try to extract leading uppercase letters
        letter_match = re.match(r'^([A-Z]+)', raw_answer.strip())
        if letter_match:
            return letter_match.group(1)
        # Handle Chinese True/False
        if "对" in raw_answer:
            return "A"
        if "错" in raw_answer:
            return "B"
        return raw_answer.strip()
