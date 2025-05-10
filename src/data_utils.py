import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR


def load_and_process_citibike_data(year: int) -> pd.DataFrame:
    """
    Load and preprocess raw Citi Bike data (from CSVs saved in RAW_DATA_DIR).

    Args:
        year: int (e.g., 2014)

    Returns:
        pd.DataFrame with cleaned trip start time and start station name
    """
    all_files = sorted(Path(RAW_DATA_DIR).glob(f"{year}-*.csv"))
    if not all_files:
        raise FileNotFoundError("No CSV files found for the specified year.")

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
        df = df.dropna(subset=['starttime', 'start_station_name'])
        df_list.append(df[['starttime', 'start_station_name']])

    df_combined = pd.concat(df_list).reset_index(drop=True)
    return df_combined


def transform_to_hourly_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw start times into hourly trip counts per station.

    Args:
        df: DataFrame with ['starttime', 'start_station_name']

    Returns:
        Hourly aggregated time-series DataFrame
    """
    df['pickup_hour'] = df['starttime'].dt.floor('h')
    grouped = df.groupby(['pickup_hour', 'start_station_name']).size().reset_index(name='rides')
    return fill_missing_rides_full_range(grouped, 'pickup_hour', 'start_station_name', 'rides')


def fill_missing_rides_full_range(df, hour_col, station_col, rides_col):
    """
    Fills in missing hours and station entries with 0 rides.

    Returns:
        DataFrame with all combinations filled
    """
    df[hour_col] = pd.to_datetime(df[hour_col])
    all_hours = pd.date_range(df[hour_col].min(), df[hour_col].max(), freq="h")
    all_stations = df[station_col].unique()

    complete_grid = pd.DataFrame(
        [(h, s) for h in all_hours for s in all_stations],
        columns=[hour_col, station_col]
    )

    merged = complete_grid.merge(df, on=[hour_col, station_col], how="left")
    merged[rides_col] = merged[rides_col].fillna(0).astype(int)
    return merged


def sliding_window_features(
    df: pd.DataFrame, feature_col="rides", window_size=12, step_size=1
) -> pd.DataFrame:
    """
    Create lag features per station using sliding windows.

    Returns:
        DataFrame with time-lagged features, target, station, and timestamp
    """
    stations = df["start_station_name"].unique()
    all_data = []

    for station in stations:
        station_df = df[df["start_station_name"] == station].reset_index(drop=True)
        values = station_df[feature_col].values
        timestamps = station_df["pickup_hour"].values

        if len(values) <= window_size:
            continue

        rows = []
        for i in range(0, len(values) - window_size, step_size):
            features = values[i:i + window_size]
            target = values[i + window_size]
            time = timestamps[i + window_size]
            rows.append(np.append(features, [target, station, time]))

        if rows:
            columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            df_transformed = pd.DataFrame(rows, columns=columns + ["target", "start_station_name", "pickup_hour"])
            all_data.append(df_transformed)

    return pd.concat(all_data, ignore_index=True)


def split_ts_data(
    df: pd.DataFrame,
    cutoff: datetime,
    target_col="target"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split time-series dataset into train/test using a date cutoff.

    Returns:
        X_train, y_train, X_test, y_test
    """
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    train = df[df["pickup_hour"] < cutoff].reset_index(drop=True)
    test = df[df["pickup_hour"] >= cutoff].reset_index(drop=True)

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test