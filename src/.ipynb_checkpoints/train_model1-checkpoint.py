# train_model.py
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from utils import haversine_km, aggregate_features_for_point, create_labels_for_grid, fetch_usgs_feed

# --- CONFIGURATION ---
CSV_PATH = "data/global_earthquakes_2015_2024.csv"  # modifie si besoin
MODEL_OUT = "models/xgb_earthquake_prob.pkl"
SCALER_OUT = "models/scaler.save"
# paramètres du problème
LOOKBACK_DAYS = 90         # fenêtre historique pour calcul des features
HORIZON_DAYS = 30          # horizon pour prédiction (ce que l'on veut prédire)
RADIUS_KM = 100            # rayon autour du point pour considérer les événements
MAG_THRESH = 5.0           # magnitude seuil pour considérer comme 'évènement significatif'
# grille d'exemple : points réguliers (tu peux charger des points réels: villes, secteurs, etc.)
LAT_MIN, LAT_MAX = -60, 60
LON_MIN, LON_MAX = -180, 180
GRID_LAT_STEPS = 20
GRID_LON_STEPS = 40

# --- CHARGER LE CSV ---
df = pd.read_csv(CSV_PATH)
# ASSUMPTION: colonnes 'time', 'latitude', 'longitude', 'mag', 'depth'
# conversion time en datetime si nécessaire
if df['time'].dtype == 'int64' or df['time'].dtype == 'float64':
    # si timestamp ms
    df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
else:
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time','latitude','longitude','mag']).reset_index(drop=True)
df = df.sort_values('time').reset_index(drop=True)

# --- Création d'une grille (exemple simple) ---
lats = np.linspace(LAT_MIN, LAT_MAX, GRID_LAT_STEPS)
lons = np.linspace(LON_MIN, LON_MAX, GRID_LON_STEPS)
grid = [(float(lat), float(lon)) for lat in lats for lon in lons]

# --- Pour chaque point de la grille, calculer features à partir des événements récents ---
X_rows = []
# On fixe 'now' à la fin du dataset pour training simplifié
global_now = df['time'].max()
events_until_now = df[df['time'] <= global_now]
for lat, lon in grid:
    # subset events in lookback window
    start_time = global_now - pd.Timedelta(days=LOOKBACK_DAYS)
    events_window = events_until_now[(events_until_now['time'] >= start_time) & (events_until_now['time'] <= global_now)].copy()
    feats = aggregate_features_for_point(events_window, lat, lon, lookback_days=LOOKBACK_DAYS, radius_km=RADIUS_KM)
    feats['lat'] = lat
    feats['lon'] = lon
    X_rows.append(feats)
X = pd.DataFrame(X_rows)

# --- Labels : ici on utilise la même timeline simplifiée (pour un vrai training faire sliding windows) ---
labels_df = create_labels_for_grid(df, grid, horizon_days=HORIZON_DAYS, radius_km=RADIUS_KM, mag_thresh=MAG_THRESH)
# joindre
data = X.merge(labels_df[['lat','lon','label']], on=['lat','lon'])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# features / target
feature_cols = ['count_events','max_mag','mean_mag','median_mag','std_mag','time_since_last_event_days','mean_depth','energy_sum','lat','lon']
Xf = data[feature_cols].fillna(0.0)
y = data['label'].astype(int)

# Optionnel : équilibrer classes si déséquilibre fort
pos_ratio = y.mean()
print(f"Positive ratio: {pos_ratio:.4f} (may need resampling)")

# --- Train/test split temporel ou simple split (ici random split pour exempler) ---
X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2, random_state=42, stratify=y)

# --- Scaler ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Entraînement XGBoost ---
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --- Évaluation ---
y_prob = model.predict_proba(X_test_scaled)[:,1]
auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')
brier = brier_score_loss(y_test, y_prob)
print(f"AUC: {auc:.4f}, Brier score: {brier:.4f}")

# Sauvegarder modèle et scaler
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"Model saved to {MODEL_OUT}, scaler saved to {SCALER_OUT}")
