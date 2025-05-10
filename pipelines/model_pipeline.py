import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib
import hopsworks
import mlflow
import dagshub

# -----------------------------
# Step 1: Connect to Hopsworks & MLflow
# -----------------------------
project = hopsworks.login(project="CDA500FINAL")
fs = project.get_feature_store()
mr = project.get_model_registry()

# DagsHub + MLflow
dagshub.init(repo_owner="dsapthavarni", repo_name="CDA500BIKE", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/dsapthavarni/CDA500BIKE.mlflow")
mlflow.set_experiment("citi-bike-trip-prediction")

# -----------------------------
# Step 2: Load lagged features
# -----------------------------
fg = fs.get_feature_group("citibike_daily_lagged", version=1)
df = fg.read()
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# Step 3: Prepare data
# -----------------------------
feature_cols = [f'lag_{i}' for i in range(1, 29)]
target_col = 'trip_count'

X = df[feature_cols]
y = df[target_col]
cutoff = df['date'].max() - pd.Timedelta(days=14)

X_train = X[df['date'] <= cutoff]
X_test = X[df['date'] > cutoff]
y_train = y[df['date'] <= cutoff]
y_test = y[df['date'] > cutoff]

# -----------------------------
# Step 4: Train and log model
# -----------------------------
with mlflow.start_run(run_name="lightgbm_28lags_final"):
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Log to MLflow
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("features_used", 28)
    mlflow.log_metric("mae", mae)

    print(f"✅ Model trained. MAE: {mae:.3f}")

    # Save model locally
    joblib.dump(model, "best_model.pkl")

    # -----------------------------
    # Step 5: Register model in Hopsworks
    # -----------------------------
    model_obj = mr.python.create_model(
        name="citibike_predictor",
        metrics={"mae": float(mae)},
        description="LightGBM with 28 lag features for Citi Bike trip prediction",
        input_example=X_test.head(1)
    )

    model_obj.save("best_model.pkl")
    print("✅ Model registered and saved to Hopsworks.")