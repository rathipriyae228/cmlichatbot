from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ✅ Load knowledge base
with open('chatbot_knowledge.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# ✅ Initialize semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')
kb_questions = [item['question'].strip().lower() for item in knowledge_base]
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)

# ✅ Semantic matching function
def find_answer_semantic(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, kb_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_match_idx].item()
    print(f"🧠 Semantic Similarity Score: {best_score:.2f}")

    if best_score > 0.6:
        best_item = knowledge_base[best_match_idx]
        response = best_item['answer']
        if best_item.get('link'):
            response += f'<br><a href="{best_item["link"]}" target="_blank">Click here for more info</a>'
        return response
    return None  # semantic failed

# ✅ TF-IDF fallback (optional backup)
def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    keywords = [feature_names[i] for i in scores.argsort()[-5:][::-1]]
    return keywords

def find_answer_tfidf(keywords):
    for item in knowledge_base:
        entry_keywords = [kw.strip().lower() for kw in item.get('keywords', [])]
        for keyword in keywords:
            if keyword.lower() in entry_keywords:
                response = item['answer']
                if item.get('link'):
                    response += f'<br><a href="{item["link"]}" target="_blank">Click here for more info</a>'
                return response
    return None

# ✅ Log unmatched queries
def log_unanswered_question(query):
    with open("unanswered.log", "a", encoding="utf-8") as f:
        f.write(query + "\n")

# ✅ Routes
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

        # Try semantic match first
        answer = find_answer_semantic(user_input)

        # If semantic fails, try TF-IDF fallback
        if not answer:
            keywords = extract_keywords(user_input)
            print(f"🔎 Extracted Keywords (TF-IDF Fallback): {keywords}")
            answer = find_answer_tfidf(keywords)

        # If still no match
        if not answer:
            log_unanswered_question(user_input)
            answer = "I'm not sure how to answer that. Please try rephrasing your question."

        return jsonify({"response": answer})

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return jsonify({"response": "An internal error occurred."})

if __name__ == '__main__':
    app.run(debug=True)
