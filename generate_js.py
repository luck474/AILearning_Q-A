import json
import os

def generate_js():
    courses = []
    
    # Define courses and their source files
    course_definitions = [
        {"id": "deep_learning", "name": "深度学习 (Deep Learning)", "file": "deep_learning.json"},
        {"id": "javaee", "name": "JavaEE (Spring/Boot/MVC)", "file": "javaee.json"}
    ]
    
    for c_def in course_definitions:
        if os.path.exists(c_def["file"]):
            with open(c_def["file"], "r", encoding="utf-8") as f:
                qs = json.load(f)
                courses.append({
                    "id": c_def["id"],
                    "name": c_def["name"],
                    "questions": qs
                })
        else:
            print(f"Warning: File {c_def['file']} not found.")

    # Output JS
    js_content = "const courses = " + json.dumps(courses, indent=2, ensure_ascii=False) + ";"
    with open("questions.js", "w", encoding="utf-8") as f:
        f.write(js_content)
    
    print(f"Generated questions.js with {len(courses)} courses.")

if __name__ == "__main__":
    generate_js()
