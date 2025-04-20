from model_training import model_push, scaler_push
from insights import generate_insights, generate_overall_insight

sample_input = {
    "user_id": 1,
    "sleep_minutes_ma7": 350,  # 7 heures -> 420 minutes
    "calories_ma7": 2500,
    "proteines_ma7": 150,
    "glucides_ma7": 350,
    "lipides_ma7": 80,
    "tonnage_last_same_type": 8000
}

insights = generate_insights(sample_input, model_push, scaler_push, variations=[0.10, 0.20])
for feat, var_data in insights.items():
    for var_label, rec in var_data.items():
        print(rec["recommendation"])

overall_message = generate_overall_insight(insights)
print("\nInsight global :")
print(overall_message)
