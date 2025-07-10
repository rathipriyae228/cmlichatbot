from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ✅ Load knowledge base
with open('chatbot_knowledge.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# ✅ TF-IDF keyword extractor
def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    keywords = [feature_names[i] for i in scores.argsort()[-5:][::-1]]
    return keywords

# ✅ Match response by keyword
def find_answer(keywords):
    for item in knowledge_base:
        entry_keywords = [kw.strip().lower() for kw in item['keywords']]
        for keyword in keywords:
            if keyword.lower() in entry_keywords:
                response = item['answer']
                if item.get('link'):
                    response += f'<br><a href="{item["link"]}" target="_blank">Click here for more info</a>'
                return response
    return "Sorry, I couldn't find an answer to your question."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/default-message', methods=['GET'])
def default_message():
    return jsonify({"response": "Hi! I'm the CMLI chatbot. Ask me anything about our research!"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "").strip().lower()
        if not user_input:
            return jsonify({"response": "Please provide a valid message."})

        keywords = extract_keywords(user_input)
        print(f"🔎 Extracted Keywords: {keywords}")

        answer = find_answer(keywords)
        return jsonify({"response": answer})

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return jsonify({"response": "An internal error occurred."})

if __name__ == '__main__':
    app.run(debug=True)
