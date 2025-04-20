import numpy as np
import pandas as pd

# Les features doivent correspondre exactement à celles utilisées lors de l'entraînement.
# Ici, nous utilisons "tonnage_last_same_type" car c'est le nom qui a été utilisé pour entraîner le scaler et le modèle.
features = [
    "user_id", 
    "sleep_minutes_ma7", 
    "calories_ma7", 
    "proteines_ma7", 
    "glucides_ma7", 
    "lipides_ma7", 
    "tonnage_last_same_type"
]

def adjust_nutrition(current_features, variation):
    """
    Ajuste de manière conjointe les calories ainsi que les macronutriments en tenant compte de la contrainte énergétique.
    
    Args:
        current_features (dict): Dictionnaire contenant les valeurs actuelles des variables nutritionnelles.
            Exemple:
            {
              "calories_ma7": 2800,
              "proteines_ma7": 150,
              "glucides_ma7": 350,
              "lipides_ma7": 80
            }
        variation (float): Pourcentage d'augmentation (exemple: 0.10 pour +10%).
    
    Returns:
        dict: Dictionnaire avec les nouvelles valeurs pour calories_ma7, proteines_ma7, glucides_ma7 et lipides_ma7.
    """
    curr_cal = current_features["calories_ma7"]
    curr_prot = current_features["proteines_ma7"]
    curr_gluc = current_features["glucides_ma7"]
    curr_lip  = current_features["lipides_ma7"]
    
    # On calcule les ratios en supposant que "calories_ma7" est cohérent avec les macros.
    ratio_prot = (4 * curr_prot) / curr_cal
    ratio_gluc = (4 * curr_gluc) / curr_cal
    ratio_lip  = (9 * curr_lip)  / curr_cal
    
    target_cal = curr_cal * (1 + variation)
    
    new_prot = target_cal * ratio_prot / 4
    new_gluc = target_cal * ratio_gluc / 4
    new_lip  = target_cal * ratio_lip  / 9
    
    return {
        "calories_ma7": target_cal,
        "proteines_ma7": new_prot,
        "glucides_ma7": new_gluc,
        "lipides_ma7": new_lip
    }

def generate_insights(current_features, model, scaler, variations=[0.10]):
    """
    Génère des recommandations pour chaque variable modifiable.
    
    Pour le sommeil, une simulation isolée est réalisée.
    Pour la nutrition (calories et macronutriments), la simulation est réalisée de manière conjointe via adjust_nutrition.
    
    Args:
        current_features (dict): Dictionnaire des valeurs actuelles non normalisées.
            Exemple :
            {
              "user_id": 3,
              "sleep_minutes_ma7": 420,
              "calories_ma7": 2800,
              "proteines_ma7": 150,
              "glucides_ma7": 350,
              "lipides_ma7": 80,
              "tonnage_last_same_type": 8000
            }
        model: le modèle de régression entraîné.
        scaler: le StandardScaler déjà ajusté sur les données d'entraînement.
        variations (list): Liste des pourcentages de variation à simuler (exemple: [0.10, 0.20]).
    
    Returns:
        dict: Dictionnaire regroupant les insights pour le sommeil et pour la nutrition.
    """
    base_df = pd.DataFrame([current_features], columns=features)
    base_scaled = scaler.transform(base_df)
    baseline_pred = model.predict(base_scaled)[0]
    
    insights = {}
    
    # Simulation isolée pour le sommeil
    insights["sleep_minutes_ma7"] = {}
    for var in variations:
        modified = current_features.copy()
        new_value = modified["sleep_minutes_ma7"] * (1 + var)
        increase = new_value - modified["sleep_minutes_ma7"]
        modified["sleep_minutes_ma7"] = new_value
        
        mod_df = pd.DataFrame([modified], columns=features)
        mod_scaled = scaler.transform(mod_df)
        new_pred = model.predict(mod_scaled)[0]
        delta = new_pred - baseline_pred
        
        insights["sleep_minutes_ma7"][f"{int(var*100)}%"] = {
            "baseline_prediction": baseline_pred,
            "new_prediction": new_pred,
            "delta": delta,
            "absolute_increase": increase,
            "recommendation": (
                f"Pour augmenter vos minutes de sommeil, vous devriez ajouter {increase:.2f} minutes "
                f"(passer de {current_features['sleep_minutes_ma7']:.2f} à {new_value:.2f}). "
                f"Cela pourrait modifier la performance prédite de {delta:.0f} kg."
            )
        }
    
    # Simulation conjointe pour la nutrition
    insights["nutrition"] = {}
    for var in variations:
        modified = current_features.copy()
        adjusted = adjust_nutrition(current_features, var)
        modified.update(adjusted)
        increase_cal = adjusted["calories_ma7"] - current_features["calories_ma7"]
        increase_prot = adjusted["proteines_ma7"] - current_features["proteines_ma7"]
        increase_gluc = adjusted["glucides_ma7"] - current_features["glucides_ma7"]
        increase_lip  = adjusted["lipides_ma7"] - current_features["lipides_ma7"]
        
        mod_df = pd.DataFrame([modified], columns=features)
        mod_scaled = scaler.transform(mod_df)
        new_pred = model.predict(mod_scaled)[0]
        delta = new_pred - baseline_pred
        
        cal_from_macros = 4 * increase_prot + 4 * increase_gluc + 9 * increase_lip
        
        insights["nutrition"][f"{int(var*100)}%"] = {
            "baseline_prediction": baseline_pred,
            "new_prediction": new_pred,
            "delta": delta,
            "absolute_increase_calories": increase_cal,
            "absolute_increase_proteines": increase_prot,
            "absolute_increase_glucides": increase_gluc,
            "absolute_increase_lipides": increase_lip,
            "cal_from_macros": cal_from_macros,
            "recommendation": (
                f"Pour augmenter vos calories de manière cohérente, vous devriez augmenter vos calories de {increase_cal:.2f} "
                f"(passer de {current_features['calories_ma7']:.2f} à {adjusted['calories_ma7']:.2f}) en ajustant simultanément "
                f"vos protéines de {increase_prot:.2f}g, vos glucides de {increase_gluc:.2f}g et vos lipides de {increase_lip:.2f}g "
                f"(ce qui représente environ {cal_from_macros:.2f} calories issues des macronutriments). "
                f"Cela pourrait modifier la performance prédite de {delta:.0f} kg."
            )
        }
    
    return insights

def generate_overall_insight(insights_dict):
    """
    Génère un message global qui combine la recommandation sur le sommeil et celle sur la nutrition.
    Pour la nutrition, le message agrège la suggestion conjointe (calories et macronutriments).
    """
    recommendations = []
    
    # Recommandation pour le sommeil
    if "sleep_minutes_ma7" in insights_dict:
        best_sleep = max(insights_dict["sleep_minutes_ma7"].values(), key=lambda x: x["delta"])
        recommendations.append(f"augmenter vos minutes de sommeil de {best_sleep['absolute_increase']:.2f}")
    
    # Recommandation pour la nutrition
    if "nutrition" in insights_dict:
        best_nutrition = max(insights_dict["nutrition"].values(), key=lambda x: x["delta"])
        recommendations.append(
            f"ajuster votre nutrition en augmentant vos calories de {best_nutrition['absolute_increase_calories']:.2f} calories "
            f"(ce qui correspond à une augmentation de {best_nutrition['absolute_increase_proteines']:.2f}g de protéines, "
            f"{best_nutrition['absolute_increase_glucides']:.2f}g de glucides et {best_nutrition['absolute_increase_lipides']:.2f}g de lipides)"
        )
        
    if recommendations:
        overall_message = "Pour améliorer vos performances, vous devriez " + ", ".join(recommendations) + "."
    else:
        overall_message = "Aucune modification significative à recommander."
    return overall_message

if __name__ == "__main__":
    sample_input = {
        "user_id": 3,
        "sleep_minutes_ma7": 420,      # 7 heures converties en minutes
        "calories_ma7": 2800,
        "proteines_ma7": 150,
        "glucides_ma7": 350,
        "lipides_ma7": 80,
        "tonnage_last_same_type": 8000
    }
    from model_training import model_push, scaler_push
    insights = generate_insights(sample_input, model_push, scaler_push, variations=[0.10, 0.20])
    for feat, var_data in insights.items():
        for var_label, rec in var_data.items():
            print(rec["recommendation"])
    overall_message = generate_overall_insight(insights)
    print("\nInsight global :")
    print(overall_message)
import numpy as np
import pandas as pd

# Les features doivent correspondre exactement à celles utilisées lors de l'entraînement.
# Ici, nous utilisons "tonnage_last_same_type" car c'est le nom qui a été utilisé pour entraîner le scaler et le modèle.
features = [
    "user_id", 
    "sleep_minutes_ma7", 
    "calories_ma7", 
    "proteines_ma7", 
    "glucides_ma7", 
    "lipides_ma7", 
    "tonnage_last_same_type"
]

def adjust_nutrition(current_features, variation):
    """
    Ajuste de manière conjointe les calories ainsi que les macronutriments en tenant compte de la contrainte énergétique.
    
    Args:
        current_features (dict): Dictionnaire contenant les valeurs actuelles des variables nutritionnelles.
            Exemple:
            {
              "calories_ma7": 2800,
              "proteines_ma7": 150,
              "glucides_ma7": 350,
              "lipides_ma7": 80
            }
        variation (float): Pourcentage d'augmentation (exemple: 0.10 pour +10%).
    
    Returns:
        dict: Dictionnaire avec les nouvelles valeurs pour calories_ma7, proteines_ma7, glucides_ma7 et lipides_ma7.
    """
    curr_cal = current_features["calories_ma7"]
    curr_prot = current_features["proteines_ma7"]
    curr_gluc = current_features["glucides_ma7"]
    curr_lip  = current_features["lipides_ma7"]
    
    # On calcule les ratios en supposant que "calories_ma7" est cohérent avec les macros.
    ratio_prot = (4 * curr_prot) / curr_cal
    ratio_gluc = (4 * curr_gluc) / curr_cal
    ratio_lip  = (9 * curr_lip)  / curr_cal
    
    target_cal = curr_cal * (1 + variation)
    
    new_prot = target_cal * ratio_prot / 4
    new_gluc = target_cal * ratio_gluc / 4
    new_lip  = target_cal * ratio_lip  / 9
    
    return {
        "calories_ma7": target_cal,
        "proteines_ma7": new_prot,
        "glucides_ma7": new_gluc,
        "lipides_ma7": new_lip
    }

def generate_insights(current_features, model, scaler, variations=[0.10]):
    """
    Génère des recommandations pour chaque variable modifiable.
    
    Pour le sommeil, une simulation isolée est réalisée.
    Pour la nutrition (calories et macronutriments), la simulation est réalisée de manière conjointe via adjust_nutrition.
    
    Args:
        current_features (dict): Dictionnaire des valeurs actuelles non normalisées.
            Exemple :
            {
              "user_id": 3,
              "sleep_minutes_ma7": 420,
              "calories_ma7": 2800,
              "proteines_ma7": 150,
              "glucides_ma7": 350,
              "lipides_ma7": 80,
              "tonnage_last_same_type": 8000
            }
        model: le modèle de régression entraîné.
        scaler: le StandardScaler déjà ajusté sur les données d'entraînement.
        variations (list): Liste des pourcentages de variation à simuler (exemple: [0.10, 0.20]).
    
    Returns:
        dict: Dictionnaire regroupant les insights pour le sommeil et pour la nutrition.
    """
    base_df = pd.DataFrame([current_features], columns=features)
    base_scaled = scaler.transform(base_df)
    baseline_pred = model.predict(base_scaled)[0]
    
    insights = {}
    
    # Simulation isolée pour le sommeil
    insights["sleep_minutes_ma7"] = {}
    for var in variations:
        modified = current_features.copy()
        new_value = modified["sleep_minutes_ma7"] * (1 + var)
        increase = new_value - modified["sleep_minutes_ma7"]
        modified["sleep_minutes_ma7"] = new_value
        
        mod_df = pd.DataFrame([modified], columns=features)
        mod_scaled = scaler.transform(mod_df)
        new_pred = model.predict(mod_scaled)[0]
        delta = new_pred - baseline_pred
        
        insights["sleep_minutes_ma7"][f"{int(var*100)}%"] = {
            "baseline_prediction": baseline_pred,
            "new_prediction": new_pred,
            "delta": delta,
            "absolute_increase": increase,
            "recommendation": (
                f"Pour augmenter vos minutes de sommeil, vous devriez ajouter {increase:.2f} minutes "
                f"(passer de {current_features['sleep_minutes_ma7']:.2f} à {new_value:.2f}). "
                f"Cela pourrait modifier la performance prédite de {delta:.0f} kg."
            )
        }
    
    # Simulation conjointe pour la nutrition
    insights["nutrition"] = {}
    for var in variations:
        modified = current_features.copy()
        adjusted = adjust_nutrition(current_features, var)
        modified.update(adjusted)
        increase_cal = adjusted["calories_ma7"] - current_features["calories_ma7"]
        increase_prot = adjusted["proteines_ma7"] - current_features["proteines_ma7"]
        increase_gluc = adjusted["glucides_ma7"] - current_features["glucides_ma7"]
        increase_lip  = adjusted["lipides_ma7"] - current_features["lipides_ma7"]
        
        mod_df = pd.DataFrame([modified], columns=features)
        mod_scaled = scaler.transform(mod_df)
        new_pred = model.predict(mod_scaled)[0]
        delta = new_pred - baseline_pred
        
        cal_from_macros = 4 * increase_prot + 4 * increase_gluc + 9 * increase_lip
        
        insights["nutrition"][f"{int(var*100)}%"] = {
            "baseline_prediction": baseline_pred,
            "new_prediction": new_pred,
            "delta": delta,
            "absolute_increase_calories": increase_cal,
            "absolute_increase_proteines": increase_prot,
            "absolute_increase_glucides": increase_gluc,
            "absolute_increase_lipides": increase_lip,
            "cal_from_macros": cal_from_macros,
            "recommendation": (
                f"Pour augmenter vos calories de manière cohérente, vous devriez augmenter vos calories de {increase_cal:.2f} "
                f"(passer de {current_features['calories_ma7']:.2f} à {adjusted['calories_ma7']:.2f}) en ajustant simultanément "
                f"vos protéines de {increase_prot:.2f}g, vos glucides de {increase_gluc:.2f}g et vos lipides de {increase_lip:.2f}g "
                f"(ce qui représente environ {cal_from_macros:.2f} calories issues des macronutriments). "
                f"Cela pourrait modifier la performance prédite de {delta:.0f} kg."
            )
        }
    
    return insights

def generate_overall_insight(insights_dict):
    """
    Génère un message global qui combine la recommandation sur le sommeil et celle sur la nutrition.
    Pour la nutrition, le message agrège la suggestion conjointe (calories et macronutriments).
    """
    recommendations = []
    
    # Recommandation pour le sommeil
    if "sleep_minutes_ma7" in insights_dict:
        best_sleep = max(insights_dict["sleep_minutes_ma7"].values(), key=lambda x: x["delta"])
        recommendations.append(f"augmenter vos minutes de sommeil de {best_sleep['absolute_increase']:.2f}")
    
    # Recommandation pour la nutrition
    if "nutrition" in insights_dict:
        best_nutrition = max(insights_dict["nutrition"].values(), key=lambda x: x["delta"])
        recommendations.append(
            f"ajuster votre nutrition en augmentant vos calories de {best_nutrition['absolute_increase_calories']:.2f} calories "
            f"(ce qui correspond à une augmentation de {best_nutrition['absolute_increase_proteines']:.2f}g de protéines, "
            f"{best_nutrition['absolute_increase_glucides']:.2f}g de glucides et {best_nutrition['absolute_increase_lipides']:.2f}g de lipides)"
        )
        
    if recommendations:
        overall_message = "Pour améliorer vos performances, vous devriez " + ", ".join(recommendations) + "."
    else:
        overall_message = "Aucune modification significative à recommander."
    return overall_message

if __name__ == "__main__":
    sample_input = {
        "user_id": 3,
        "sleep_minutes_ma7": 420,      # 7 heures converties en minutes
        "calories_ma7": 2800,
        "proteines_ma7": 150,
        "glucides_ma7": 350,
        "lipides_ma7": 80,
        "tonnage_last_same_type": 8000
    }
    from model_training import model_push, scaler_push
    insights = generate_insights(sample_input, model_push, scaler_push, variations=[0.10, 0.20])
    for feat, var_data in insights.items():
        for var_label, rec in var_data.items():
            print(rec["recommendation"])
    overall_message = generate_overall_insight(insights)
    print("\nInsight global :")
    print(overall_message)
