import pandas as pd
import hopsworks

# -----------------------------
# Step 1: Connect to Hopsworks
# -----------------------------
project = hopsworks.login(
    project=os.environ["CDA500FINAL"],
    api_key_value=os.environ["HB0zAW5eEzl4iuNq.KJX5bZAdAnGaRJrIFVFVB30exr8wMMql5TZUuNMVeMUbcOVqRXg0fW3OWz2aRzOi"]
)
fs = project.get_feature_store()

# -----------------------------
# Step 2: Load top 3 station data
# -----------------------------
fg_raw = fs.get_feature_group("citibike_2014_top3", version=1)
df = fg_raw.read()

# -----------------------------
# Step 3: Prepare daily trip counts
# -----------------------------
df['starttime'] = pd.to_datetime(df['starttime'])
df['date'] = df['starttime'].dt.date

daily_counts = (
    df.groupby(['start_station_name', 'date'])
    .size()
    .reset_index(name='trip_count')
    .sort_values(['start_station_name', 'date'])
)

# -----------------------------
# Step 4: Create 28-day lag features
# -----------------------------
for lag in range(1, 29):
    daily_counts[f'lag_{lag}'] = (
        daily_counts.groupby('start_station_name')['trip_count'].shift(lag)
    )

# Drop rows with incomplete lag history
daily_lagged = daily_counts.dropna().reset_index(drop=True)

# Add timestamp column for Hopsworks event_time
daily_lagged['date'] = pd.to_datetime(daily_lagged['date'])

# -----------------------------
# Step 5: Write to Hopsworks Feature Store
# -----------------------------
fg_lagged = fs.get_or_create_feature_group(
    name="citibike_daily_lagged",
    version=1,
    primary_key=["date", "start_station_name"],
    event_time="date",
    description="Daily trip counts with 28 lag features for top 3 stations"
)

fg_lagged.insert(daily_lagged, write_options={"wait_for_job": True})

print("✅ Feature engineering complete and stored in Hopsworks.")
