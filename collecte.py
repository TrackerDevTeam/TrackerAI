# %% Importer les bibliothèques 
import os
import json
import pandas as pd
import numpy as np
from pandasgui import show
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -------------------------------------------------------------------------
# Fonction pour charger les données
# -------------------------------------------------------------------------
def charger_donnees(base_path="data"):
    """
    Charge récursivement toutes les données des utilisateurs stockées sous `base_path`
    et les structure dans un dictionnaire.
    """
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
                                # Stocke les données avec la clé du nom du fichier sans l'extension
                                all_data[user_id][date_folder][file_name.split('.')[0]] = json.load(f)
    return all_data

# -------------------------------------------------------------------------
# Chargement et transformation des données
# -------------------------------------------------------------------------
donnees = charger_donnees("data")

data_nutrition = []
data_sommeil = []
data_training = []

for user, dates in donnees.items():
    for date, files in dates.items():
        if "nutrition" in files:
            record = files["nutrition"]
            record["user"], record["date"] = user, date
            data_nutrition.append(record)
        if "sommeil" in files:
            record = files["sommeil"]
            record["user"], record["date"] = user, date
            data_sommeil.append(record)
        if "training" in files:
            record = files["training"]
            record["user"], record["date"] = user, date
            data_training.append(record)

df_nutrition = pd.DataFrame(data_nutrition)
df_sommeil = pd.DataFrame(data_sommeil)
df_training = pd.DataFrame(data_training)

# -------------------------------------------------------------------------
# Amélioration du DataFrame Sommeil (pour cohérence)
# -------------------------------------------------------------------------
def convertir_total_heure(heure_str):
    """Convertit un format '7h38' en minutes."""
    if isinstance(heure_str, str) and "h" in heure_str:
        heures, minutes = map(int, heure_str.split("h"))
        return heures * 60 + minutes
    return None

# Normalisation du format de données dans sommeil
df_sommeil_expanded = pd.json_normalize(df_sommeil.loc[:, 'sommeil'])
df_sommeil_clean = pd.concat([
    df_sommeil[['user', 'date']].reset_index(drop=True), 
    df_sommeil_expanded.reset_index(drop=True)
], axis=1)
df_sommeil_clean['duree_minutes'] = df_sommeil_clean['total_heure'].apply(convertir_total_heure)

# -------------------------------------------------------------------------
# Création du DataFrame Exercices
# -------------------------------------------------------------------------
liste_exercices = []
for _, row in df_training.iterrows():
    user, date = row["user"], row["date"]
    exercices = row.get("exercices", [])
    for ex in exercices:
        ex_record = {"user": user, "date": date}
        ex_record.update(ex)
        liste_exercices.append(ex_record)

df_exercices = pd.DataFrame(liste_exercices)

# -------------------------------------------------------------------------
# Fusion des DataFrames dans df_global (pour cohérence)
# -------------------------------------------------------------------------
df_global = pd.merge(df_training, df_nutrition, on=["user", "date"], how="inner")
df_global = pd.merge(df_global, df_sommeil_clean, on=["user", "date"], how="inner")

# Affichage interactif via pandasgui (optionnel)
print("\nDataFrames Disponibles :")
print(f"- df_nutrition : {df_nutrition.shape}")
print(f"- df_sommeil : {df_sommeil_clean.shape}")
print(f"- df_training : {df_training.shape}")
print(f"- df_exercices : {df_exercices.shape}")
print(f"- df_global : {df_global.shape}")
gui = show(df_global)

# -------------------------------------------------------------------------
# 1. Statistiques Descriptives
# -------------------------------------------------------------------------
print("\nStatistiques Descriptives:")

# Statistiques pour les variables d'entraînement présentes dans df_training
stats_training = df_training.describe()
print("\nStatistiques pour les données d'entraînement:")
print(stats_training)

# Statistiques pour la nutrition
stats_nutrition = df_nutrition.describe()
print("\nStatistiques pour les données de nutrition:")
print(stats_nutrition)

# Statistiques pour le sommeil
stats_sommeil = df_sommeil_clean.describe()
print("\nStatistiques pour les données de sommeil:")
print(stats_sommeil)

# Exemple : Afficher quelques statistiques spécifiques
variables_a_etudier = ['duree_minutes', 'nombre_series_total', 'calories']
for var in variables_a_etudier:
    if var in df_global.columns:
        print(f"\nStatistiques pour {var} dans df_global:")
        print(df_global[var].describe())

# -------------------------------------------------------------------------
# 2. Visualisations
# -------------------------------------------------------------------------

## 2a. Distribution des variables

# Histogramme et Boxplot pour la durée d'entraînement
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_training["duree_minutes"], bins=20, kde=True)
plt.title("Histogramme de la durée d'entraînement (minutes)")
plt.subplot(1, 2, 2)
sns.boxplot(x=df_training["duree_minutes"])
plt.title("Boxplot de la durée d'entraînement")
plt.show()

# Histogramme et Boxplot pour les calories
if 'calories' in df_global.columns:
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df_global["calories"], bins=20, kde=True)
    plt.title("Histogramme des calories consommées")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_global["calories"])
    plt.title("Boxplot des calories consommées")
    plt.show()

# Histogramme et Boxplot pour la qualité du sommeil (si variable catégorielle convertie ou en numérique)
# Ici on peut par exemple mapper la qualité en numérique si possible :
if 'qualite' in df_sommeil_clean.columns:
    qual_mapping = {"faible": 1, "moyenne": 2, "bonne": 3}  # adapter selon vos données
    df_sommeil_clean["qualite_num"] = df_sommeil_clean["qualite"].map(qual_mapping)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df_sommeil_clean["qualite_num"], bins=3, kde=False)
    plt.title("Histogramme de la qualité du sommeil (numérique)")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_sommeil_clean["qualite_num"])
    plt.title("Boxplot de la qualité du sommeil")
    plt.show()

## 2b. Visualisation des corrélations entre variables

# Sélectionner quelques variables pertinentes provenant de df_global
variables_corr = ["duree_minutes", "nombre_series_total", "tonnage_total", "calories", "proteines", "glucides", "lipides"]

# Vérifier si ces variables sont dans df_global et construire la matrice de corrélation
variables_existantes = [var for var in variables_corr if var in df_global.columns]
corr_matrix = df_global[variables_existantes].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap des corrélations entre variables")
plt.show()

## 2c. Scatterplots pour explorer les relations

# Relation entre la durée d'entraînement et les calories consommées
if "duree_minutes" in df_global.columns and "calories" in df_global.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_global, x="duree_minutes", y="calories", hue="user", palette="tab10")
    plt.title("Relation entre la durée d'entraînement et les calories")
    plt.xlabel("Durée d'entraînement (minutes)")
    plt.ylabel("Calories consommées")
    plt.show()

# Relation entre l'apport en protéines et le tonnage total (performance d'entraînement)
if "proteines" in df_global.columns and "tonnage_total" in df_global.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_global, x="proteines", y="tonnage_total", hue="user", palette="tab10")
    plt.title("Relation entre l'apport en protéines et le tonnage total")
    plt.xlabel("Protéines (g)")
    plt.ylabel("Tonnage total (kg)")
    plt.show()

# -------------------------------------------------------------------------
# 3. Analyse de l'évolution des performances dans le temps pour chaque utilisateur
# -------------------------------------------------------------------------

# Exemple avec l'évolution du tonnage total dans df_training
df_training['date'] = pd.to_datetime(df_training['date'], format="%Y_%m_%d")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_training, x="date", y="tonnage_total", hue="user", marker="o")
plt.title("Évolution du tonnage total par session d'entraînement")
plt.xlabel("Date")
plt.ylabel("Tonnage total (kg)")
plt.xticks(rotation=45)
plt.show()

# -------------------------------------------------------------------------
# 4. Evolution du poids du squat pour tous les utilisateurs avec régression linéaire
# -------------------------------------------------------------------------
# Filtrer les enregistrements concernant l'exercice "squat" (insensible à la casse)
df_squat = df_exercices[df_exercices["nom"].str.lower() == "squat"].copy()

# Conversion de la date en datetime et création d'une colonne numérique pour la régression
df_squat["date"] = pd.to_datetime(df_squat["date"], format="%Y_%m_%d")
df_squat["date_ord"] = df_squat["date"].apply(lambda d: d.toordinal())

# Palette pour associer chaque utilisateur à une couleur
unique_users = df_squat["user"].unique()
palette = dict(zip(unique_users, sns.color_palette("tab10", n_colors=len(unique_users))))

# Affichage avec lmplot pour la régression
plot = sns.lmplot(
    x="date_ord", 
    y="poids", 
    hue="user", 
    data=df_squat, 
    markers="o", 
    scatter_kws={'s': 50}, 
    ci=None,      # Supprime l'intervalle de confiance
    height=5, 
    aspect=1.5,
    legend=True,
    robust=True  # Optionnel, pour réduire l'influence des outliers
)

ax = plot.ax

# Utilisation d'un locateur et formateur pour fixer le nombre de ticks et formater les dates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Ajout des équations de régression pour chaque utilisateur après le nom
for user in unique_users:
    df_user = df_squat[df_squat["user"] == user]
    if len(df_user) >= 2:
        # Calcul de la régression linéaire (coefficients)
        a, b = np.polyfit(df_user["date_ord"], df_user["poids"], 1)
        poly = np.poly1d((a, b))
        # Positionnement de l'annotation en utilisant la date maximale
        x_max_ord = int(df_user["date_ord"].max())
        y_max = poly(x_max_ord)
        x_max = pd.Timestamp.fromordinal(x_max_ord)
        # Affichage : nom de l'utilisateur suivi de l'équation
        eq_text = f"{user} (y = {a:.2f}x + {b:.2f})"
        ax.text(x_max, y_max, eq_text, color=palette[user],
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor=palette[user]))

ax.set_xlabel("Date")
ax.set_ylabel("Poids utilisé pour le squat (kg)")
plt.title("Évolution du poids du squat avec régression linéaire pour tous les utilisateurs")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# 5. Agrégations temporelles (rolling, hebdo, mensuelles)
# -------------------------------------------------------------------------
df_global['date'] = pd.to_datetime(df_global['date'], format="%Y_%m_%d")
df_global.sort_values(['user', 'date'], inplace=True)

# Moyenne mobile sur 7 jours pour le tonnage total
df_global['tonnage_moy_7j'] = df_global.groupby('user')['tonnage_total'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Agrégation hebdomadaire (par user)
df_weekly = df_global.copy()
df_weekly['week'] = df_weekly['date'].dt.to_period('W').apply(lambda r: r.start_time)
df_weekly_agg = df_weekly.groupby(['user', 'week'])[['tonnage_total', 'calories', 'duree_minutes']].mean().reset_index()
df_weekly_agg.rename(columns={
    'tonnage_total': 'tonnage_moy_hebdo',
    'calories': 'calories_moy_hebdo',
    'duree_minutes': 'duree_moy_hebdo'
}, inplace=True)

# Agrégation mensuelle (par user)
df_monthly = df_global.copy()
df_monthly['month'] = df_monthly['date'].dt.to_period('M').astype(str)
df_monthly_agg = df_monthly.groupby(['user', 'month'])[['tonnage_total', 'calories', 'duree_minutes']].mean().reset_index()
df_monthly_agg.rename(columns={
    'tonnage_total': 'tonnage_moy_mensuel',
    'calories': 'calories_moy_mensuel',
    'duree_minutes': 'duree_moy_mensuelle'
}, inplace=True)

# -------------------------------------------------------------------------
# 6. Indicateurs de progression d’un jour à l’autre
# -------------------------------------------------------------------------
df_global['tonnage_ratio_jour_prec'] = df_global.groupby('user')['tonnage_total'].pct_change()
df_global['duree_ratio_jour_prec'] = df_global.groupby('user')['duree_minutes'].pct_change()
df_global['calories_ratio_jour_prec'] = df_global.groupby('user')['calories'].pct_change()

# -------------------------------------------------------------------------
# 7. Normalisation des variables (utile pour un futur modèle ML)
# -------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Variables à normaliser
cols_to_normalize = ['tonnage_total', 'duree_minutes', 'calories', 'proteines', 'glucides', 'lipides']

# Supprimer les lignes avec NaN pour l'entraînement éventuel
df_norm_input = df_global[cols_to_normalize].dropna()

# Normalisation Min-Max
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df_norm_input), columns=[f"{col}_minmax" for col in cols_to_normalize])

# Standardisation (z-score)
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df_norm_input), columns=[f"{col}_zscore" for col in cols_to_normalize])

# Concat avec les infos user/date (si besoin pour entraînement)
df_norm_final = pd.concat([
    df_global[['user', 'date']].iloc[df_norm_input.index].reset_index(drop=True),
    df_minmax.reset_index(drop=True),
    df_std.reset_index(drop=True)
], axis=1)

# -------------------------------------------------------------------------
# 8. Création d’un label supervisé : "progrès" vs "stagnation/régression"
# -------------------------------------------------------------------------
def classifier_progression(x):
    if pd.isna(x):
        return None
    elif x > 0.05:
        return "progrès"
    elif x < -0.05:
        return "régression"
    else:
        return "stable"

df_global['label_progression'] = df_global['tonnage_ratio_jour_prec'].apply(classifier_progression)
