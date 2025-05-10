from datetime import datetime, timedelta, timezone
from pathlib import Path

import hopsworks
import joblib
import pandas as pd

import src.config as config
from src.data_utils import transform_ts_data_info_features


def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )


def get_feature_store():
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(features)
    return pd.DataFrame({
        "start_station_name": features["start_station_name"].values,
        "predicted_demand": predictions.round(0)
    })


def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    fs = get_feature_store()

    fetch_to = current_date - timedelta(hours=1)
    fetch_from = current_date - timedelta(days=29)

    feature_view = fs.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION,
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_from - timedelta(days=1)),
        end_time=(fetch_to + timedelta(days=1)),
    )

    ts_data = ts_data[ts_data.pickup_hour.between(fetch_from, fetch_to)]
    ts_data.sort_values(by=["start_station_name", "pickup_hour"], inplace=True)

    features = transform_ts_data_info_features(
        ts_data, feature_col="rides", window_size=24 * 28, step_size=23
    )
    return features


def load_model_from_registry(version=None):
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)
    best_model = max(models, key=lambda m: m.version)
    model_dir = best_model.download()

    return joblib.load(Path(model_dir) / "lgb_model.pkl")


def load_metrics_from_registry(version=None):
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)
    best_model = max(models, key=lambda m: m.version)

    return best_model.training_metrics


def fetch_next_hour_predictions():
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_PREDICTION_NAME, version=1)
    df = fg.read()
    return df[df["pickup_hour"] == next_hour]


def fetch_predictions(hours=1):
    current_hour = (pd.Timestamp.now(tz="UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_PREDICTION_NAME, version=1)

    df = fg.filter((fg.pickup_hour >= current_hour)).read()
    return df


def fetch_hourly_rides(hours=1):
    current_hour = (pd.Timestamp.now(tz="UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all().filter(fg.pickup_hour >= current_hour)
    return query.read()


def fetch_days_data(days=1):
    current_date = pd.to_datetime(datetime.now(timezone.utc))
    fetch_from = current_date - timedelta(days=(365 + days))
    fetch_to = current_date - timedelta(days=365)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    df = fg.select_all().read()
    return df[(df["pickup_hour"] >= fetch_from) & (df["pickup_hour"] <= fetch_to)]