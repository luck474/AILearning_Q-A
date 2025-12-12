import re
import json
import os

def parse_javaee_questions(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    questions = []
    current_q = None
    
    # regex patterns
    # Start of a question: "1. (判断题)..." or "1. (单选题)..."
    q_start_pattern = re.compile(r'^\d+\.\s*\((.+?)\)(.*)')
    # Option pattern: "A. ...", "B. ...", "A、...", "B、..."
    option_pattern = re.compile(r'^([A-Z])[\.、]\s*(.*)')
    # Answer pattern: "正确答案[:：](.*)"
    answer_pattern = re.compile(r'.*正确答案[:：](.*)')
    
    # Store lines that belong to the current question context until next question starts
    # But the file has sections like "一. ...". We should ignore those or use them as separators.
    
    # Improved strategy: Iterate line by line.
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a section header (e.g. "一. 单选题...")
        # Dictionary keys usually don't start with Chinese numbers like "一.", but questions start with arabic "1."
        if re.match(r'^[一二三四五六七八九十]+\.', line):
            continue
            
        q_match = q_start_pattern.match(line)
        if q_match:
            # Save previous question if exists
            if current_q:
                questions.append(current_q)
            
            # Start new question
            q_type_str = q_match.group(1).strip()
            q_text = q_match.group(2).strip()
            
            # Map type
            mapped_type = "unknown"
            if "判断题" in q_type_str:
                mapped_type = "true_false"
            elif "单选题" in q_type_str:
                mapped_type = "multiple_choice"
            elif "多选题" in q_type_str:
                mapped_type = "multiple_response"
            elif "填空题" in q_type_str:
                mapped_type = "fill_in_the_blank" 
            
            current_q = {
                "id": len(questions) + 1,
                "question": q_text,
                "type": mapped_type,
                "options": [],
                "answer": "",
                "raw_answer": "" # Temp storage
            }
            
            # For True/False, we can pre-populate options or wait for "A. 对" lines
            if mapped_type == "true_false":
                # Some T/F questions might not list options explicitly in text, but usually they do.
                # If they don't, we should add default T/F options?
                # Looking at file: "A. 对", "B. 错" are present.
                pass
                
        elif current_q:
            # Processing lines for current question
            
            # Check for Options
            opt_match = option_pattern.match(line)
            if opt_match:
                # raw option line
                # "A. 对" -> "A. 对"
                # Store full line as option to preserve "A." prefix which is often used in display
                current_q["options"].append(line)
                continue
                
            # Check for Answer
            ans_match = answer_pattern.match(line)
            if ans_match:
                raw_ans = ans_match.group(1).strip()
                current_q["raw_answer"] = raw_ans
                # Process answer
                # "对" -> "A" (if A is 对), "错" -> "B"
                # "ABCDE:..." -> "ABCDE"
                # "D:/math/arith/calcu;" -> "D"
                
                # Simple extraction of leading letters or Chinese chars
                
                # Case 1: Multiple choice/response with letters "ABCDE:..."
                # Extract starting uppercase letters
                letter_match = re.match(r'^([A-Z]+)', raw_ans)
                if letter_match:
                     current_q["answer"] = letter_match.group(1)
                else:
                    # Case 2: T/F with "对" or "错"
                    if "对" in raw_ans:
                        current_q["answer"] = "A" # Assumes A is True/对
                        # Verify options if possible, but usually A=True
                    elif "错" in raw_ans:
                        current_q["answer"] = "B" # Assumes B is False/错
                continue

            # Append other text to question if it looks like part of question (and not score/analysis)
            # Skip scoring lines "10分", "4.1分"
            if "分" in line and len(line) < 10:
                continue
            # Skip "AI讲解", "知识点", "我的答案" (if not containing correct answer)
            if line.startswith("AI讲解") or line.startswith("知识点") or line.startswith("我的答案"):
                continue
            
            # If it's just text and we haven't seen options yet, append to question
            if not current_q["options"] and not current_q["answer"]:
                current_q["question"] += "\n" + line

    # Append last question
    if current_q:
        questions.append(current_q)
        
    # Filter and clean up
    final_questions = []
    for q in questions:
        # Skip unsupported types
        if q["type"] == "fill_in_the_blank" or q["type"] == "unknown":
            continue
            
        # Clean answer
        # If true_false, ensure answer is normalized? 
        # The frontend `checkCorrectness` for true_false expects "TRUE" or "FALSE" or "A"/"B"?
        # Script.js: `return selected.toUpperCase() === answer.toString().toUpperCase();`
        # And options for T/F in this file are "A. 对", "B. 错".
        # So if user clicks "A. 对", selected is "A. 对".
        # The script extracts key: `match = selected.match(/^([A-D])[\.、]/); key = match ? match[1] : selected;`
        # So key will be "A" or "B".
        # So we should store answer as "A" or "B" for T/F too?
        # script.js `if (type === 'true_false') { return selected.toUpperCase() === answer.toString().toUpperCase(); }`
        # Wait, for true_false, script.js Logic is:
        # `return selected.toUpperCase() === answer.toString().toUpperCase();`
        # But if options are "A. 对", selected is "A. 对". `selected.toUpperCase()` is "A. 对".
        # If answer is "True", "A. 对" != "True".
        # Let's check script.js line 218 again.
        # `checkCorrectness` for true_false.
        # It seems the previous dataset had options ["True", "False"] and answer "True".
        # Now we have "A. 对", "B. 错".
        # We should treat them as multiple_choice logic basically, or update script.js.
        # EASIEST FIX: Change type to `multiple_choice` for T/F questions in JSON, 
        # OR keep type `true_false` but set options ["A. 对", "B. 错"] and answer "A".
        # AND update script.js to handle T/F like multiple choice if it detects letters?
        # Actually checking line 225 in script.js:
        # `const match = selected.match(/^([A-D])[\.、]/);`
        # This is in `else` block (Single Choice).
        # Section `if (type === 'true_false')` is separate.
        # I should just map T/F to `multiple_choice` to avoid logic issues in frontend, 
        # since they are structurally identical (pick one of two options).
        
        if q["type"] == "true_false":
            q["type"] = "multiple_choice" # Treat as multiple choice A/B
            
        final_questions.append(q)
        
    # Re-ID
    for i, q in enumerate(final_questions):
        q["id"] = i + 1

    return final_questions

if __name__ == "__main__":
    qs = parse_javaee_questions("JavaEE题库.txt")
    
    # Backup existing
    if os.path.exists("questions.json"):
        os.rename("questions.json", "questions_backup.json")
        
    with open("questions.json", "w", encoding="utf-8") as f:
        json.dump(qs, f, indent=2, ensure_ascii=False)
        
    print(f"Imported {len(qs)} questions.")
