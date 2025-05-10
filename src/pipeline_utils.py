import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# -----------------------------
# Feature: Average Rides Last 4 Weeks
# -----------------------------
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    weekly_lag_cols = [
        f"rides_t-{7*24}",   # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]

    for col in weekly_lag_cols:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    X["average_rides_last_4_weeks"] = X[weekly_lag_cols].mean(axis=1)
    return X


add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# -----------------------------
# Feature: Temporal Info (hour of day, day of week)
# -----------------------------
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["hour"] = X["pickup_hour"].dt.hour
        X["day_of_week"] = X["pickup_hour"].dt.dayofweek
        return X.drop(columns=["pickup_hour", "start_station_name"])


add_temporal_features = TemporalFeatureEngineer()


# -----------------------------
# Final Modeling Pipeline
# -----------------------------
def get_pipeline(**hyper_params):
    """
    Returns a full modeling pipeline with preprocessing + LightGBM.

    Parameters:
        hyper_params (dict): Parameters for LGBMRegressor

    Returns:
        pipeline (Pipeline): scikit-learn pipeline
    """
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params)
    )