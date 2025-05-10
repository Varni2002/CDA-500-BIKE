from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: str,
    predictions: Optional[pd.Series] = None,
):
    """
    Plots time series for a specific station using historical features and actual/predicted values.

    Args:
        features (pd.DataFrame): Feature data (including lagged rides and pickup_hour).
        targets (pd.Series): Actual target values (ride demand).
        row_id (str): Start station name for the plot.
        predictions (Optional[pd.Series]): Optional prediction series for the same station.

    Returns:
        plotly.graph_objects.Figure
    """
    # Filter for the specific station
    station_features = features[features["start_station_name"] == row_id].iloc[0]
    actual_target = targets[features["start_station_name"] == row_id].values[0]

    # Extract lag features
    lag_cols = [col for col in features.columns if col.startswith("rides_t-")]
    lag_values = station_features[lag_cols].values.tolist()

    # Generate time range
    pickup_hour = pd.to_datetime(station_features["pickup_hour"])
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(lag_cols)),
        periods=len(lag_cols) + 1,
        freq="H"
    )

    # Combine historical and target values
    values = lag_values + [actual_target]

    fig = go.Figure()

    # Historical + actual
    fig.add_trace(go.Scatter(
        x=time_series_dates,
        y=values,
        mode="lines+markers",
        name="Actual",
        line=dict(color="green"),
        marker=dict(size=6)
    ))

    # Optional: prediction marker
    if predictions is not None:
        predicted_value = predictions[features["start_station_name"] == row_id].values[0]
        fig.add_trace(go.Scatter(
            x=[pickup_hour],
            y=[predicted_value],
            mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            name="Prediction"
        ))

    # Title and labels
    fig.update_layout(
        title=f"⏱️ Station: {row_id} | Hour: {pickup_hour}",
        xaxis_title="Time",
        yaxis_title="Ride Counts",
        template="plotly_white"
    )

    return fig


def plot_prediction(features: pd.DataFrame, prediction: pd.DataFrame):
    """
    Plots past demand and next-hour prediction for a given station (row 0 assumed).

    Args:
        features (pd.DataFrame): Feature DataFrame with lag columns and pickup_hour
        prediction (pd.DataFrame): Prediction DataFrame with 'predicted_demand'

    Returns:
        plotly.graph_objects.Figure
    """
    lag_cols = [col for col in features.columns if col.startswith("rides_t-")]
    lag_values = features.iloc[0][lag_cols].values.tolist()
    pickup_hour = pd.to_datetime(features["pickup_hour"].iloc[0])
    station = features["start_station_name"].iloc[0]
    predicted_value = prediction["predicted_demand"].iloc[0]

    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(lag_cols)),
        periods=len(lag_cols) + 1,
        freq="H"
    )
    values = lag_values + [predicted_value]

    fig = px.line(
        x=time_series_dates,
        y=values,
        markers=True,
        template="plotly_white",
        title=f"📍 {station} | Prediction for {pickup_hour}",
        labels={"x": "Time", "y": "Ride Counts"},
    )

    fig.add_scatter(
        x=[pickup_hour],
        y=[predicted_value],
        mode="markers",
        marker=dict(color="red", size=10, symbol="x"),
        name="Prediction"
    )

    return fig