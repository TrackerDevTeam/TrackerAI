# app.py
from flask import Flask, jsonify
from model_training import model_push, scaler_push
from insights import generate_insights, generate_overall_insight

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route("/insight", methods=["GET"])
def get_insight():
    sample_input = {
        "user_id": 3,
        "sleep_minutes_ma7": 420,      # 7 heures converties en minutes
        "calories_ma7": 2800,
        "proteines_ma7": 150,
        "glucides_ma7": 350,
        "lipides_ma7": 80,
        "tonnage_last_same_type": 8000
    }
    insights_dict = generate_insights(sample_input, model_push, scaler_push, variations=[0.10, 0.20])
    message = generate_overall_insight(insights_dict)
    return jsonify({"insight": message})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
