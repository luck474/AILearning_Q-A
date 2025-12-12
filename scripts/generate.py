#!/usr/bin/env python3
"""
Generate the frontend questions.js file from all course data.

Usage:
    python3 scripts/generate.py
"""
import json
import os
from config import COURSES, get_data_path, get_output_path


def generate_questions_js():
    """Generate the questions.js file for the frontend."""
    courses_data = []
    
    for course in COURSES:
        data_path = get_data_path(course["data_file"])
        
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping {course['id']}")
            continue
        
        with open(data_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        courses_data.append({
            "id": course["id"],
            "name": course["name"],
            "questions": questions
        })
    
    # Generate JS output
    output_path = get_output_path("questions.js")
    js_content = "const courses = " + json.dumps(courses_data, indent=2, ensure_ascii=False) + ";"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    
    print(f"Generated {output_path} with {len(courses_data)} courses.")


if __name__ == "__main__":
    generate_questions_js()
