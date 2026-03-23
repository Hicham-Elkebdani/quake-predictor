# utils.py
import math
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

def haversine_km(lat1, lon1, lat2, lon2):
    # distance en km
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def fetch_usgs_feed(feed='all_day'):
    """
    Récupère le GeoJSON public de l'USGS.
    feed: 'all_hour', 'all_day', 'all_week', 'all_month' ou summary ones (e.g. '4.5_day')
    retourne pandas.DataFrame avec colonnes ['time','latitude','longitude','magnitude','depth_km','id']
    """
    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    rows = []
    for f in data.get('features', []):
        p = f.get('properties', {})
        g = f.get('geometry', {})
        coords = g.get('coordinates') or [None, None, None]
        ts = p.get('time')  # ms since epoch
        try:
            t = pd.to_datetime(ts, unit='ms')
        except:
            t = None
        rows.append({
            'time': t,
            'latitude': coords[1],
            'longitude': coords[0],
            'depth_km': coords[2] if len(coords) > 2 else None,
            'magnitude': p.get('magnitude'),
            'id': f.get('id')
        })
    return pd.DataFrame(rows)

def aggregate_features_for_point(events_df, center_lat, center_lon, lookback_days=30, radius_km=100):
    """
    Calcule des features simples pour la position (center_lat, center_lon) en regardant
    les événements dans les lookback_days passés et dans radius_km.
    events_df doit contenir: time (datetime), latitude, longitude, magnitude, depth_km
    Retourne dict de features.
    """
    now = events_df['time'].max() if 'time' in events_df.columns else pd.Timestamp.utcnow()
    start_time = now - pd.Timedelta(days=lookback_days)
    mask_time = (events_df['time'] >= start_time) & (events_df['time'] <= now)
    recent = events_df.loc[mask_time].copy()
    if recent.empty:
        # features vides
        return {
            'count_events': 0,
            'max_mag': 0.0,
            'mean_mag': 0.0,
            'median_mag': 0.0,
            'std_mag': 0.0,
            'time_since_last_event_days': float('inf'),
            'mean_depth': 0.0,
            'energy_sum': 0.0
        }
    # compute distances
    recent['distance_km'] = recent.apply(
        lambda r: haversine_km(center_lat, center_lon, r['latitude'], r['longitude']), axis=1
    )
    within = recent[recent['distance_km'] <= radius_km]
    if within.empty:
        return {
            'count_events': 0,
            'max_mag': 0.0,
            'mean_mag': 0.0,
            'median_mag': 0.0,
            'std_mag': 0.0,
            'time_since_last_event_days': float('inf'),
            'mean_depth': 0.0,
            'energy_sum': 0.0
        }
    count = len(within)
    max_mag = within['magnitude'].max()
    mean_mag = within['magnitude'].mean()
    median_mag = within['magnitude'].median()
    std_mag = within['magnitude'].std(ddof=0) if count > 1 else 0.0
    last_time = within['time'].max()
    tsince = (now - last_time) / pd.Timedelta(days=1)
    mean_depth = within['depth_km'].mean()
    # approximate seismic energy from magnitude (log scale): E ~ 10^(1.5*M)
    energy_sum = (10 ** (1.5 * within['magnitude'].clip(lower=0))).sum()
    return {
        'count_events': int(count),
        'max_mag': float(max_mag) if not pd.isna(max_mag) else 0.0,
        'mean_mag': float(mean_mag) if not pd.isna(mean_mag) else 0.0,
        'median_mag': float(median_mag) if not pd.isna(median_mag) else 0.0,
        'std_mag': float(std_mag) if not pd.isna(std_mag) else 0.0,
        'time_since_last_event_days': float(tsince) if not pd.isna(tsince) else float('inf'),
        'mean_depth': float(mean_depth) if not pd.isna(mean_depth) else 0.0,
        'energy_sum': float(energy_sum) if not pd.isna(energy_sum) else 0.0
    }

def create_labels_for_grid(events_df, grid_points, horizon_days=30, radius_km=50, mag_thresh=5.0):
    """
    Pour chaque point de la grille (liste de (lat, lon)), crée un label 1/0:
    1 s'il existe un événement de magnitude >= mag_thresh dans radius_km
    dans l'intervalle (t, t + horizon_days) pour t = each sample time (futur)
    Ici on fait simplification: on prendra comme 'sample time' la fin d'une fenêtre d'observation (events_df).
    Retourne DataFrame labels: columns = ['lat','lon','label','event_time','event_mag']
    """
    labels = []
    # on suppose events_df contient événements historiques ; on utilisera sliding windows si besoin.
    # simplification : pour chaque grid point, label = 1 s'il y a au moins un événement mag>=mag_thresh dans horizon_days
    # NOTE: pour training réel il faut créer de multiples samples chronologiques.
    max_time = events_df['time'].max()
    end_window = max_time
    start_window = end_window - pd.Timedelta(days=horizon_days)
    future_mask = (events_df['time'] > end_window) & (events_df['time'] <= end_window + pd.Timedelta(days=horizon_days))
    future_events = events_df.loc[future_mask]
    for lat, lon in grid_points:
        label = 0
        found = None
        for _, ev in future_events.iterrows():
            d = haversine_km(lat, lon, ev['latitude'], ev['longitude'])
            if (d <= radius_km) and (ev['magnitude'] is not None) and (ev['magnitude'] >= mag_thresh):
                label = 1
                found = ev
                break
        labels.append({
            'lat': lat,
            'lon': lon,
            'label': label,
            'event_time': found['time'] if found is not None else None,
            'event_mag': float(found['magnitude']) if found is not None else None
        })
    return pd.DataFrame(labels)
