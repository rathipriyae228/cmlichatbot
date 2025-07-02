from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yake
import json
import requests

# ✅ Flask App Configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ✅ JSON Source from GitHub (make sure this is the RAW link)
JSON_URL = "https://raw.githubusercontent.com/Chellamsreec/cmli/main/chatbot_knowledge.json"

# ✅ Load JSON data from GitHub at startup
try:
    response = requests.get(JSON_URL)
    response.raise_for_status()
    knowledge_base = response.json()
    print("✅ JSON knowledge base loaded from GitHub")
except Exception as e:
    print(f"❌ Failed to load knowledge base from GitHub: {e}")
    knowledge_base = []

# ✅ YAKE Keyword Extractor
keyword_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.6, top=5)

# ✅ Homepage Route
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Initial Greeting
@app.route('/default-message', methods=['GET'])
def default_message():
    return jsonify({"response": "Hello! How can I help you?"})

# ✅ Chat Route with Accuracy Metrics
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "").strip().lower()
        if not user_input:
            return jsonify({"response": "Please provide a valid message."})

        # Extract keywords
        extracted_keywords = [kw[0].lower().strip() for kw in keyword_extractor.extract_keywords(user_input)]
        print(f"🔎 Extracted Keywords: {extracted_keywords}")

        # Search in knowledge base
        for item in knowledge_base:
            expected_keywords = item.get("keywords", [])
            if any(kw in expected_keywords for kw in extracted_keywords):
                response = item["answer"]
                if item.get("link"):
                    response += f'<br><a href="{item["link"]}" target="_blank">Click here for more info</a>'

                # ✅ Accuracy Metrics
                tp = set(extracted_keywords) & set(expected_keywords)
                precision = len(tp) / len(extracted_keywords) if extracted_keywords else 0
                recall = len(tp) / len(expected_keywords) if expected_keywords else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

                # ✅ Print to console
                print(f"✅ Expected Keywords: {expected_keywords}")
                print(f"🎯 True Positives: {list(tp)}")
                print(f"📊 Precision: {round(precision, 2)} | Recall: {round(recall, 2)} | F1 Score: {round(f1, 2)}")

                return jsonify({"response": response})

        return jsonify({"response": "Sorry, I couldn't find an answer for that."})

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return jsonify({"response": "An error occurred. Please try again."})

# ✅ Run App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
