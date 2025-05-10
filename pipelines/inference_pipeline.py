import pandas as pd
import hopsworks
import joblib
from datetime import timedelta

# -----------------------------
# Step 1: Connect to Hopsworks
# -----------------------------
project = hopsworks.login(project="CDA500FINAL")
fs = project.get_feature_store()
mr = project.get_model_registry()

# -----------------------------
# Step 2: Load latest lag features
# -----------------------------
fg_lagged = fs.get_feature_group("citibike_daily_lagged", version=1)
df = fg_lagged.read()
df['date'] = pd.to_datetime(df['date'])

# Get the most recent day's features
latest_date = df['date'].max()
today_data = df[df['date'] == latest_date]

# -----------------------------
# Step 3: Load registered model
# -----------------------------
model = mr.get_model("citibike_predictor", version=1)
model_dir = model.download()
model = joblib.load(f"{model_dir}/best_model.pkl")

# -----------------------------
# Step 4: Predict next-day trip counts
# -----------------------------
feature_cols = [f'lag_{i}' for i in range(1, 29)]
X_today = today_data[feature_cols]
predictions = model.predict(X_today)

# Create prediction DataFrame
next_day = latest_date + timedelta(days=1)
prediction_df = pd.DataFrame({
    "date": next_day,
    "start_station_name": today_data["start_station_name"].values,
    "predicted_trip_count": predictions
})

# -----------------------------
# Step 5: Insert predictions into Hopsworks
# -----------------------------
fg_pred = fs.get_or_create_feature_group(
    name="citibike_predictions",
    version=1,
    primary_key=["date", "start_station_name"],
    event_time="date",
    description="Predicted trip counts for next day using LightGBM"
)

fg_pred.insert(prediction_df, write_options={"wait_for_job": True})
print("✅ Inference complete. Predictions saved to Hopsworks.")