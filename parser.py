import re
import json

def parse_questions(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines or lines that look like start of new questions
    # The file has questions numbered like "1: ..." or just text.
    # Looking at the file content provided in history:
    # 1: ...
    # 2: T
    # 3: F
    #
    # 5: ...
    # 6: A...
    
    # It seems blocks are separated by empty lines.
    blocks = content.split('\n\n')
    
    questions = []
    
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
            
        question_text = lines[0]
        # Remove leading "123: " if present
        question_text = re.sub(r'^\d+\s*[:\.]\s*', '', question_text)
        
        options = []
        q_type = "unknown"
        
        # Check for options
        cleaned_lines = lines[1:]
        
        # Check if it's T/F
        if len(cleaned_lines) >= 1 and (cleaned_lines[0].upper() == 'T' or cleaned_lines[0].startswith('T ')):
             q_type = "true_false"
             options = ["True", "False"]
        # Check if it's Multiple Choice (starts with A. B. etc)
        elif any(l.startswith('A.') or l.startswith('A、') for l in cleaned_lines):
            q_type = "multiple_choice"
            current_option = ""
            for l in cleaned_lines:
                # Basic parsing for A. ... B. ... lines
                # Sometimes options are on separate lines, sometimes not.
                # In this file, they seem to be on separate lines usually: "6: A. ..."
                # remove line numbers like "6: "
                l = re.sub(r'^\d+\s*[:\.]\s*', '', l)
                options.append(l)
        else:
             # Fallback: just add remaining lines as potential context or options
             pass

        questions.append({
            "id": len(questions) + 1,
            "question": question_text,
            "type": q_type,
            "options": options,
            "answer": "" # Placeholder
        })
        
    return questions

if __name__ == "__main__":
    qs = parse_questions("题库.txt")
    with open("questions.json", "w", encoding="utf-8") as f:
        json.dump(qs, f, indent=2, ensure_ascii=False)
