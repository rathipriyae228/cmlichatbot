from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import requests
import torch

# Flask setup
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load knowledge base
JSON_URL = "https://raw.githubusercontent.com/Chellamsreec/cmli/main/chatbot_knowledge.json"
try:
    response = requests.get(JSON_URL)
    response.raise_for_status()
    knowledge_base = response.json()
    print("✅ Knowledge base loaded.")
except Exception as e:
    print(f"❌ Failed to load knowledge base: {e}")
    knowledge_base = []

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SentenceTransformer loaded.")

# Precompute question embeddings
questions = [item["question"] for item in knowledge_base]
question_embeddings = model.encode(questions, convert_to_tensor=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "Please enter a question."})

        input_embedding = model.encode(user_input, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)

        # Get best match
        best_match_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[0][best_match_idx].item()

        if best_score > 0.6:  # confidence threshold
            matched = knowledge_base[best_match_idx]
            answer = matched["answer"]
            if "link" in matched:
                answer += f'<br><a href="{matched["link"]}" target="_blank">More Info</a>'
            print(f"✅ Match: {matched['question']} (Score: {round(best_score,2)})")
            return jsonify({"response": answer})
        else:
            print(f"❌ Low match score: {round(best_score, 2)}")
            return jsonify({"response": "Sorry, I couldn't understand your question. Please rephrase it."})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"response": "Something went wrong. Please try again later."})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
