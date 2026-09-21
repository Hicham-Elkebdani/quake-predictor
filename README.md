# Earthquake Risk Predictor

Earthquake Risk Predictor is a machine learning project designed to estimate the probability of a seismic event in a given region using historical earthquake data and live USGS feeds. The system uses XGBoost to combine historical patterns with recent seismic activity in order to provide a quick risk estimate for disaster preparedness and early-warning support.

## Project overview

This project aims to:

- analyze historical earthquake data from 2015 to 2024
- retrieve real-time earthquake activity from the USGS API
- compute contextual features around a point or region
- train a classification model to predict whether a risky event is likely
- expose risk estimation through a lightweight Flask API

The model focuses on features such as:

- number of recent events in a radius
- maximum and average magnitude
- standard deviation of magnitude
- time since the last event
- average depth
- cumulative seismic energy proxy

## Features

- Historical earthquake dataset processing
- Real-time USGS feed integration
- Geospatial calculations using the haversine formula
- Aggregated seismic risk features per location
- XGBoost-based classification model
- Flask API for prediction and feature extraction
- Scalable and easy-to-extend architecture

## Repository structure

```text
quake-predictor/
├── data/
│   └── Untitled.ipynb
├── models/
│   └── (trained model files generated after training)
├── src/
│   ├── app.py
│   ├── train_model1.py
│   └── utils.py
├── .gitignore
├── packages.txt
├── README.md
├── requirements.txt
└── .
```

## Tech stack

- Python 3
- Flask
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Requests
- Joblib

## Installation

Clone the repository:

```bash
git clone https://github.com/Hicham-Elkebdani/quake-predictor.git
cd quake-predictor
```

Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the model

The model is trained using the script in `src/train_model1.py`.

```bash
cd src
python train_model1.py
```

This script:

1. loads the earthquake dataset from `data/`
2. builds geospatial features around a grid of points
3. creates labels for future event risk
4. trains an XGBoost classifier
5. saves the trained model and scaler in the `models/` folder

## Running the API

Start the Flask API:

```bash
cd src
python app.py
```

The app runs by default on:

```text
http://0.0.0.0:5000
```

## API endpoints

### GET /health

Checks that the service is running.

Example response:

```json
{
  "status": "ok",
  "time": "2025-01-01T12:00:00.000000"
}
```

### POST /online_features

Calculates risk features for a given coordinates and radius.

Example request body:

```json
{
  "lat": 34.02,
  "lon": -118.25,
  "lookback_days": 90,
  "radius_km": 100,
  "usgs_feed": "all_day"
}
```

### POST /predict

Returns the earthquake risk probability for a location.

Example request body:

```json
{
  "lat": 34.02,
  "lon": -118.25,
  "lookback_days": 90,
  "radius_km": 100,
  "horizon_days": 30,
  "usgs_feed": "all_day"
}
```

Example response:

```json
{
  "lat": 34.02,
  "lon": -118.25,
  "probability": 0.73,
  "model": "xgb_earthquake_prob.pkl",
  "horizon_days": 30
}
```

## Data sources

The project uses:

- historical earthquake data from 2015–2024
- live earthquake feeds from the USGS API

The live feed is accessed at:

```text
https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson
```

## Model logic

The model uses a feature vector built from recent seismic activity near a target point, including:

- event count within a radius
- maximum magnitude
- average magnitude
- magnitude variability
- time since latest event
- average depth
- approximate seismic energy

These features are standardized before being passed to the XGBoost classifier, which outputs a probability between 0 and 1 representing the likelihood of a significant earthquake in the given region.

## Notes

- The project is a practical proof-of-concept and can be improved with more advanced temporal modeling.
- A more realistic version would use rolling time windows, additional physical features, and spatially-aware validation.
- The current repository includes an example notebook and scripts for experimentation and deployment.

## License

This repository does not currently declare a specific license. If needed, add one before public distribution or commercial use.

## Contributing

Pull requests and improvements are welcome. You can contribute by:

- improving model accuracy
- refining feature engineering
- expanding geospatial analysis
- improving API validation and documentation

## Contact

For questions or collaboration requests, please contact the repository owner or open an issue in the GitHub repository.
