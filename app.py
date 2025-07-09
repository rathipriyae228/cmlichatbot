from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yake
import mysql.connector

# ✅ Flask App Configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ✅ Database Configuration
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "Chellam@1",  # Replace with your actual MySQL password
    "database": "cmli_chatbot"
}

# ✅ Optimized YAKE Keyword Extraction Settings
custom_kw_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    dedupLim=0.6,
    top=5
)

# ✅ Function to Establish a Database Connection
def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        print(f"❌ Database Connection Error: {err}")
        return None

# ✅ Fetch Response and Expected Keywords from MySQL Using Keywords
def get_response_and_expected_keywords(keywords):
    conn = get_db_connection()
    if conn is None:
        return "Database connection failed. Please try again later.", []

    cursor = conn.cursor(dictionary=True)

    try:
        for keyword in keywords:
            print(f"🔍 Checking keyword: {keyword}")

            query = """
                SELECT response, reference_link, question_keywords 
                FROM faq 
                WHERE question_keywords LIKE CONCAT('%', %s, '%') 
                LIMIT 1
            """

            cursor.execute(query, (keyword,))
            result = cursor.fetchone()

            if result:
                response = result['response']
                link = result.get('reference_link')

                if link:
                    response += f'<br><a href="{link}" target="_blank">Click here for more info</a>'

                expected_keywords = result['question_keywords'].split(',')
                expected_keywords = [kw.strip().lower() for kw in expected_keywords]

                print(f"✅ Match Found: {response}")
                return response, expected_keywords

        print("❌ No Match Found in DB")
        return "Sorry, I couldn't find an answer to your question.", []

    except mysql.connector.Error as err:
        print(f"❌ Query Error: {err}")
        return "There was an issue fetching the response.", []

    finally:
        cursor.close()
        conn.close()

# ✅ Route to Load Chatbot UI
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Route for Initial Greeting
@app.route('/default-message', methods=['GET'])
def default_message():
    return jsonify({"response": "Hello! How can I help you?"})

# ✅ Chatbot Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "").strip().lower()

        if not user_input:
            return jsonify({"response": "Please provide a valid message."})

        # Extract keywords from user input
        extracted_keywords = [kw[0].lower().strip() for kw in custom_kw_extractor.extract_keywords(user_input)]
        print(f"🔎 Extracted Keywords: {extracted_keywords}")

        # Get matching response and expected keywords from DB
        response, expected_keywords = get_response_and_expected_keywords(extracted_keywords)

        # Optional: Still print accuracy info in terminal for debugging
        if expected_keywords:
            set_extracted = set(extracted_keywords)
            set_expected = set(expected_keywords)
            true_positives = set_extracted & set_expected

            precision = len(true_positives) / len(set_extracted) if set_extracted else 0
            recall = len(true_positives) / len(set_expected) if set_expected else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"✅ Expected Keywords: {expected_keywords}")
            print(f"🎯 True Positives: {list(true_positives)}")
            print(f"📊 Precision: {round(precision, 2)}")
            print(f"📈 Recall: {round(recall, 2)}")
            print(f"🏆 F1 Score: {round(f1, 2)}\n")
        else:
            print("⚠️ No expected keywords found for accuracy calculation.\n")

        return jsonify({"response": response})

    except Exception as e:
        print(f"❌ Error in /chat route: {e}")
        return jsonify({"response": "An error occurred while processing your request."})

# ✅ Run the App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
