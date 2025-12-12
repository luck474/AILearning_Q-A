import json

# Recovered answers from previous session
answers_key = [
    "True", "B", "C", "A", "False", "False", "A", "D", "B", "False",
    "True", "D", "A", "unknown", "unknown", "unknown",
    "B", "A", "C", "C", "C", "C", "B", "B", "C",
    "B", "B", "ABCD", "C", "C", "D", "B", "B", "B",
    "C", "B", "C", "B", "B", "B", "B", "D", "A",
    "B", "B", "C", "C", "C", "B", "B"
]

def fix_dl_answers():
    try:
        with open("deep_learning.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        updated_count = 0
        for i, q in enumerate(questions):
            if i < len(answers_key):
                ans = answers_key[i]
                if ans != "unknown":
                    q["answer"] = ans
                    updated_count += 1
            else:
                print(f"Warning: No answer found for QID {q['id']}")

        with open("deep_learning.json", "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully updated {updated_count} answers in deep_learning.json")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix_dl_answers()
