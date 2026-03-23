# app.py
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
from utils import fetch_usgs_feed, aggregate_features_for_point
import os
from datetime import datetime

MODEL_PATH = "models/xgb_earthquake_prob.pkl"
SCALER_PATH = "models/scaler.save"

app = Flask(__name__)

# Charger modèle et scaler au démarrage
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler not found. Exécute train_model.py d'abord.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "time": datetime.utcnow().isoformat()})

@app.route("/online_features", methods=["POST"])
def online_features():
    """
    Body JSON expected:
    {
      "lat": 34.02,
      "lon": -118.25,
      "lookback_days": 90,
      "radius_km": 100,
      "usgs_feed": "all_day"
    }
    """
    try:
        body = request.get_json()
        lat = float(body.get('lat'))
        lon = float(body.get('lon'))
        lookback_days = int(body.get('lookback_days', 90))
        radius_km = float(body.get('radius_km', 100))
        feed = body.get('usgs_feed', 'all_day')
        events = fetch_usgs_feed(feed=feed)
        # ensure time parsed
        events = events.dropna(subset=['time','latitude','longitude','magnitude'])
        feats = aggregate_features_for_point(events, lat, lon, lookback_days=lookback_days, radius_km=radius_km)
        return jsonify({"features": feats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    Body JSON:
    {
      "lat": 34.02,
      "lon": -118.25,
      "lookback_days": 90,         # pour calcul des features
      "radius_km": 100,
      "horizon_days": 30,          # pour info (le modèle a été entrainé pour un horizon)
      "usgs_feed": "all_day"
    }
    """
    try:
        body = request.get_json()
        lat = float(body.get('lat'))
        lon = float(body.get('lon'))
        lookback_days = int(body.get('lookback_days', 90))
        radius_km = float(body.get('radius_km', 100))
        feed = body.get('usgs_feed', 'all_day')
        events = fetch_usgs_feed(feed=feed)
        events = events.dropna(subset=['time','latitude','longitude','magnitude'])
        feats = aggregate_features_for_point(events, lat, lon, lookback_days=lookback_days, radius_km=radius_km)
        # construire DataFrame puis scaler puis prédiction
        import pandas as pd
        feature_cols = ['count_events','max_mag','mean_mag','median_mag','std_mag','time_since_last_event_days','mean_depth','energy_sum','lat','lon']
        row = {k: feats.get(k, 0.0) for k in feature_cols}
        row['lat'] = lat
        row['lon'] = lon
        X = pd.DataFrame([row])
        X_scaled = scaler.transform(X[feature_cols])
        prob = float(model.predict_proba(X_scaled)[:,1][0])
        return jsonify({
            "lat": lat,
            "lon": lon,
            "probability": prob,
            "model": os.path.basename(MODEL_PATH),
            "horizon_days": body.get('horizon_days', None)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # exécuter en dev : flask app.py
    app.run(host='0.0.0.0', port=5000, debug=True)
