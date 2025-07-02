from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yake
import difflib
import requests

# ✅ Flask Configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ✅ Knowledge Base Source (Your RAW JSON URL or Local Backup)
JSON_URL = "https://raw.githubusercontent.com/Chellamsreec/cmli/main/chatbot_knowledge.json"

try:
    response = requests.get(JSON_URL)
    response.raise_for_status()
    knowledge_base = response.json()
    print("✅ Knowledge base loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load knowledge base: {e}")
    knowledge_base = []

# ✅ YAKE Keyword Extractor
keyword_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.6, top=5)

# ✅ Fuzzy Matching Between Keywords
def is_fuzzy_match(user_keywords, expected_keywords, threshold=0.6):
    for user_kw in user_keywords:
        for expected_kw in expected_keywords:
            similarity = difflib.SequenceMatcher(None, user_kw, expected_kw).ratio()
            if similarity >= threshold:
                return True
    return False

# ✅ Suggest Closest Matching Questions (if no match found)
def get_closest_questions(user_input, top_n=3):
    user_input = user_input.lower()
    scored = []
    for item in knowledge_base:
        question = item.get("question", "")
        score = difflib.SequenceMatcher(None, user_input, question.lower()).ratio()
        scored.append((score, question, item.get("answer", "")))

    scored.sort(reverse=True)
    top_matches = scored[:top_n]
    suggestions = [f"<strong>Q:</strong> {q}<br><strong>A:</strong> {a}" for _, q, a in top_matches if _ > 0.4]
    return suggestions

# ✅ Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/default-message', methods=['GET'])
def default_message():
    return jsonify({"response": "Hello! How can I help you with CMLI-related queries?"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "").strip().lower()
        if not user_input:
            return jsonify({"response": "Please enter a question."})

        # Extract keywords
        extracted_keywords = [kw[0].lower().strip() for kw in keyword_extractor.extract_keywords(user_input)]
        print(f"🔎 Extracted Keywords: {extracted_keywords}")

        # Search knowledge base using fuzzy keyword match
        for item in knowledge_base:
            expected_keywords = [kw.lower().strip() for kw in item.get("keywords", [])]

            if is_fuzzy_match(extracted_keywords, expected_keywords):
                response = item["answer"]
                if item.get("link"):
                    response += f'<br><a href="{item["link"]}" target="_blank">More Info</a>'

                # Accuracy Metrics
                tp = set(extracted_keywords) & set(expected_keywords)
                precision = len(tp) / len(extracted_keywords) if extracted_keywords else 0
                recall = len(tp) / len(expected_keywords) if expected_keywords else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

                print(f"✅ Matched with: {expected_keywords}")
                print(f"📊 Precision: {round(precision,2)}, Recall: {round(recall,2)}, F1 Score: {round(f1,2)}")

                return jsonify({"response": response})

        # No exact match found — suggest similar questions
        suggestions = get_closest_questions(user_input)
        if suggestions:
            suggestion_block = "<br><br>Did you mean:<br>" + "<br><br>".join(suggestions)
            return jsonify({"response": f"Sorry, I couldn’t find an exact match. {suggestion_block}"})
        else:
            return jsonify({"response": "Sorry, I couldn’t understand. Please rephrase your question."})

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return jsonify({"response": "An error occurred. Please try again later."})

# ✅ Run App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
