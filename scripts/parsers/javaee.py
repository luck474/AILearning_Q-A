"""
Parser for JavaEE question bank files.
"""
import re
from typing import List, Dict, Any
from .base import BaseParser


class JavaEEParser(BaseParser):
    """Parser for JavaEE-style question bank files."""
    
    # Regex patterns
    QUESTION_START = re.compile(r'^\d+\.\s*\((.+?)\)(.*)')
    OPTION = re.compile(r'^([A-Z])[\.、]\s*(.*)')
    ANSWER = re.compile(r'.*正确答案[:：](.*)')
    SECTION_HEADER = re.compile(r'^[一二三四五六七八九十]+\.')
    
    def parse(self) -> List[Dict[str, Any]]:
        """Parse the JavaEE question file."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        raw_questions = []
        current_q = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip section headers
            if self.SECTION_HEADER.match(line):
                continue
            
            # Check for new question
            q_match = self.QUESTION_START.match(line)
            if q_match:
                if current_q:
                    raw_questions.append(current_q)
                
                q_type = self._map_type(q_match.group(1).strip())
                current_q = {
                    "question": q_match.group(2).strip(),
                    "type": q_type,
                    "options": [],
                    "raw_answer": ""
                }
                continue
            
            if not current_q:
                continue
            
            # Check for option
            if self.OPTION.match(line):
                current_q["options"].append(line)
                continue
            
            # Check for answer
            ans_match = self.ANSWER.match(line)
            if ans_match:
                current_q["raw_answer"] = ans_match.group(1).strip()
                continue
            
            # Skip metadata lines
            if self._is_metadata(line):
                continue
            
            # Append to question text if no options yet
            if not current_q["options"]:
                current_q["question"] += "\n" + line
        
        # Append last question
        if current_q:
            raw_questions.append(current_q)
        
        # Filter and finalize
        self.questions = self._finalize(raw_questions)
        return self.questions
    
    def _map_type(self, type_str: str) -> str:
        """Map Chinese type string to internal type."""
        if "判断题" in type_str:
            return self.TYPE_MULTIPLE_CHOICE  # Treat as MC for frontend compatibility
        if "单选题" in type_str:
            return self.TYPE_MULTIPLE_CHOICE
        if "多选题" in type_str:
            return self.TYPE_MULTIPLE_RESPONSE
        if "填空题" in type_str:
            return self.TYPE_FILL_BLANK
        return self.TYPE_UNKNOWN
    
    def _is_metadata(self, line: str) -> bool:
        """Check if line is metadata (scores, analysis, etc.)."""
        if "分" in line and len(line) < 10:
            return True
        return line.startswith(("AI讲解", "知识点", "我的答案"))
    
    def _finalize(self, raw_questions: List[Dict]) -> List[Dict[str, Any]]:
        """Filter unsupported types and finalize question format."""
        final = []
        for q in raw_questions:
            if q["type"] in (self.TYPE_FILL_BLANK, self.TYPE_UNKNOWN):
                continue
            
            final.append({
                "id": len(final) + 1,
                "question": q["question"],
                "type": q["type"],
                "options": q["options"],
                "answer": self.normalize_answer(q["raw_answer"])
            })
        return final
