import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor, plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Fonction pour charger les données
# -------------------------------------------------------------------------
def charger_donnees(base_path="data"):
    all_data = {}
    for user_id in os.listdir(base_path):
        user_path = os.path.join(base_path, user_id)
        if os.path.isdir(user_path):
            all_data[user_id] = {}
            for date_folder in os.listdir(user_path):
                date_path = os.path.join(user_path, date_folder)
                if os.path.isdir(date_path):
                    all_data[user_id][date_folder] = {}
                    for file_name in ["nutrition.json", "sommeil.json", "training.json"]:
                        file_path = os.path.join(date_path, file_name)
                        if os.path.exists(file_path):
                            with open(file_path, "r", encoding="utf-8") as f:
                                all_data[user_id][date_folder][file_name.split('.')[0]] = json.load(f)
    return all_data

# -------------------------------------------------------------------------
# Fonction de conversion du temps de sommeil en minutes
# -------------------------------------------------------------------------
def convertir_total_heure(heure_str):
    if not isinstance(heure_str, str):
        return None

    heure_str = heure_str.strip().lower()
    try:
        if "h" in heure_str:
            parts = heure_str.split("h")
            heures = int(parts[0].strip())
            minutes = int(parts[1].strip()) if parts[1].strip() != "" else 0
            return heures * 60 + minutes
        elif ":" in heure_str:
            parts = heure_str.split(":")
            heures = int(parts[0].strip())
            minutes = int(parts[1].strip())
            return heures * 60 + minutes
        elif "." in heure_str:
            heures = float(heure_str)
            return int(round(heures * 60))
        else:
            heures = float(heure_str)
            return int(round(heures * 60))
    except Exception:
        return None

# -------------------------------------------------------------------------
# Traitement des données de sommeil
# -------------------------------------------------------------------------
donnees = charger_donnees("data")
data_sommeil = []
for user, dates in donnees.items():
    for date, files in dates.items():
        if "sommeil" in files:
            record = files["sommeil"]
            record["user"], record["date"] = user, date
            data_sommeil.append(record)

df_sommeil = pd.DataFrame(data_sommeil)
df_sommeil_expanded = pd.json_normalize(df_sommeil.loc[:, 'sommeil'])
df_sommeil_clean = pd.concat([
    df_sommeil[['user', 'date']].reset_index(drop=True),
    df_sommeil_expanded.reset_index(drop=True)
], axis=1)

df_sommeil_clean['duree_minutes'] = df_sommeil_clean.apply(
    lambda row: convertir_total_heure(row.get("total_heure", row.get("heures", None))), axis=1)

df_sommeil_clean['user_id'] = df_sommeil_clean['user'].str.extract(r'(\d+)').astype(int)

# -------------------------------------------------------------------------
# Fonction de détection du type de séance
# -------------------------------------------------------------------------
def get_session_type(exercises):
    if not exercises:
        return "unknown"
    push_keywords = ["dc", "développé", "bench", "push"]
    pull_keywords = ["tirage", "row", "pull", "traction", "soulevé"]
    leg_keywords = ["squat", "jambe", "leg", "extension", "curl"]

    tonnage_scores = {"push": 0, "pull": 0, "leg": 0, "other": 0}
    for ex in exercises:
        name = ex.get("nom", "").lower()
        tonnage = ex.get("tonnage", 0)
        matched = False
        if any(kw in name for kw in leg_keywords):
            tonnage_scores["leg"] += tonnage
            matched = True
        if not matched and any(kw in name for kw in push_keywords):
            tonnage_scores["push"] += tonnage
            matched = True
        if not matched and any(kw in name for kw in pull_keywords):
            tonnage_scores["pull"] += tonnage
            matched = True
        if not matched:
            tonnage_scores["other"] += tonnage
    return max(tonnage_scores, key=tonnage_scores.get)

# -------------------------------------------------------------------------
# Agrégation des données pour le modèle
# -------------------------------------------------------------------------
base_dir = "./data"
records = []
for user_id in range(1, 11):
    user_folder = os.path.join(base_dir, f"user{user_id}")
    start_date = datetime(2025, 1, 1)
    for d in range(365):
        date_obj = start_date + timedelta(days=d)
        date_str = date_obj.strftime("%Y_%m_%d")
        day_folder = os.path.join(user_folder, date_str)
        if not os.path.isdir(day_folder):
            continue
        try:
            with open(os.path.join(day_folder, "sommeil.json"), "r", encoding="utf-8") as f:
                sleep_data = json.load(f)
            with open(os.path.join(day_folder, "nutrition.json"), "r", encoding="utf-8") as f:
                nutr_data = json.load(f)
            with open(os.path.join(day_folder, "training.json"), "r", encoding="utf-8") as f:
                train_data = json.load(f)
        except FileNotFoundError:
            continue

        sleep_str = sleep_data.get("total_heure", None) or sleep_data.get("heures", sleep_data.get("hours", None))
        sleep_minutes = convertir_total_heure(sleep_str) if sleep_str else None

        calories = nutr_data.get("calories", None)
        proteines = nutr_data.get("proteines", nutr_data.get("proteins", None))
        glucides = nutr_data.get("glucides", nutr_data.get("carbs", None))
        lipides = nutr_data.get("lipides", nutr_data.get("fats", None))

        exercises = train_data.get("exercices", [])
        tonnage = sum(ex.get("tonnage", 0) for ex in exercises) if exercises else 0
        session_type = get_session_type(exercises) if exercises else None

        records.append({
            "user_id": user_id,
            "date": date_str,
            "sleep_minutes": sleep_minutes,
            "calories": calories,
            "proteines": proteines,
            "glucides": glucides,
            "lipides": lipides,
            "type": session_type,
            "tonnage": tonnage
        })

df = pd.DataFrame(records)
df.sort_values(by=["user_id", "date"], inplace=True)
df = df.merge(df_sommeil_clean[['user_id', 'date', 'duree_minutes']], on=['user_id', 'date'], how='left')
df['sleep_minutes'] = df['duree_minutes']
df.drop(columns=['duree_minutes'], inplace=True)

# Moyennes mobiles sur 7 jours
rolling_cols = ["sleep_minutes", "calories", "proteines", "glucides", "lipides"]
for col in rolling_cols:
    df[f"{col}_ma7"] = df.groupby("user_id")[col].transform(lambda x: x.rolling(window=7, min_periods=7).mean())

df = df.groupby("user_id", group_keys=False).apply(lambda group: group.iloc[7:])

# Sessions filtrées
df_sessions = df[df["type"].notna() & (df["tonnage"] > 0)].copy()
df_sessions["tonnage_last_same_type"] = np.nan
last_tonnage = {user: {"push": None, "pull": None, "leg": None} for user in df["user_id"].unique()}
for idx, row in df_sessions.iterrows():
    user = row["user_id"]
    typ = row["type"]
    df_sessions.at[idx, "tonnage_last_same_type"] = last_tonnage[user][typ]
    last_tonnage[user][typ] = row["tonnage"]
df_sessions.dropna(subset=["tonnage_last_same_type"], inplace=True)

# Features et target
features = ["user_id", "sleep_minutes_ma7", "calories_ma7", "proteines_ma7", "glucides_ma7", "lipides_ma7", "tonnage_last_same_type"]
target = "tonnage"

data_push = df_sessions[df_sessions["type"] == "push"]
data_pull = df_sessions[df_sessions["type"] == "pull"]
data_leg = df_sessions[df_sessions["type"] == "leg"]

scaler_push = StandardScaler().fit(data_push[features])
scaler_pull = StandardScaler().fit(data_pull[features])
scaler_leg = StandardScaler().fit(data_leg[features])

def train_test_split_chrono(dataframe, date_col, split_date_str):
    train_df = dataframe[dataframe[date_col] < split_date_str]
    test_df = dataframe[dataframe[date_col] >= split_date_str]
    return train_df[features], test_df[features], train_df[target], test_df[target]

split_date = "2025_11_01"
X_train_push, X_test_push, y_train_push, y_test_push = train_test_split_chrono(data_push, "date", split_date)
X_train_pull, X_test_pull, y_train_pull, y_test_pull = train_test_split_chrono(data_pull, "date", split_date)
X_train_leg, X_test_leg, y_train_leg, y_test_leg = train_test_split_chrono(data_leg, "date", split_date)

X_train_push = scaler_push.transform(X_train_push)
X_test_push = scaler_push.transform(X_test_push)
X_train_pull = scaler_pull.transform(X_train_pull)
X_test_pull = scaler_pull.transform(X_test_pull)
X_train_leg = scaler_leg.transform(X_train_leg)
X_test_leg = scaler_leg.transform(X_test_leg)

model_push = XGBRegressor(objective="reg:squarederror", random_state=42).fit(X_train_push, y_train_push)
model_pull = XGBRegressor(objective="reg:squarederror", random_state=42).fit(X_train_pull, y_train_pull)
model_leg = XGBRegressor(objective="reg:squarederror", random_state=42).fit(X_train_leg, y_train_leg)

y_pred_push = model_push.predict(X_test_push)
y_pred_pull = model_pull.predict(X_test_pull)
y_pred_leg = model_leg.predict(X_test_leg)

y_test_all = np.concatenate([y_test_push, y_test_pull, y_test_leg])
y_pred_all = np.concatenate([y_pred_push, y_pred_pull, y_pred_leg])

print(f"Push - RMSE: {np.sqrt(mean_squared_error(y_test_push, y_pred_push)):.2f}, MAE: {mean_absolute_error(y_test_push, y_pred_push):.2f}")
print(f"Pull - RMSE: {np.sqrt(mean_squared_error(y_test_pull, y_pred_pull)):.2f}, MAE: {mean_absolute_error(y_test_pull, y_pred_pull):.2f}")
print(f"Leg  - RMSE: {np.sqrt(mean_squared_error(y_test_leg, y_pred_leg)):.2f}, MAE: {mean_absolute_error(y_test_leg, y_pred_leg):.2f}")
print(f"Global - RMSE: {np.sqrt(mean_squared_error(y_test_all, y_pred_all)):.2f}, MAE: {mean_absolute_error(y_test_all, y_pred_all):.2f}")

# Affichage des paramètres et importance des variables
print("\nParamètres PUSH:", model_push.get_params())
print("Paramètres PULL:", model_pull.get_params())
print("Paramètres LEG:", model_leg.get_params())

plot_importance(model_push, title="Feature importance - PUSH")
plt.show()
plot_importance(model_pull, title="Feature importance - PULL")
plt.show()
plot_importance(model_leg, title="Feature importance - LEG")
plt.show()

# Affichage arbre (texte)
print("\nPremier arbre - PUSH:\n")
print(model_push.get_booster().get_dump()[0])

if __name__ == "__main__":
    print("Entraînement terminé.")