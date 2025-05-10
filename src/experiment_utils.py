import logging
import os
import mlflow
from mlflow.models import infer_signature
from dotenv import load_dotenv

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


def set_mlflow_tracking():
    """
    Set up MLflow tracking server credentials and URI for Citi Bike project.
    Loads MLFLOW_TRACKING_URI from environment.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI not set in .env or system environment.")

    mlflow.set_tracking_uri(uri)
    logger.info(f"✅ MLflow tracking URI set to: {uri}")
    return mlflow


def log_model_to_mlflow(
    model,
    input_data,
    experiment_name="citi-bike-trip-prediction",
    metric_name="mae",
    model_name="citibike_predictor",
    params=None,
    score=None,
):
    """
    Log a trained model, parameters, and metrics to MLflow and optionally register it.

    Parameters:
    - model: Trained model object (e.g., LightGBM or sklearn model)
    - input_data: DataFrame used to train/predict (for signature)
    - experiment_name: MLflow experiment name
    - metric_name: Metric to log (default: "mae")
    - model_name: Registered model name in MLflow (default: "citibike_predictor")
    - params: Dict of hyperparameters
    - score: Metric value to log
    """
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"🎯 Experiment: {experiment_name}")

        with mlflow.start_run():
            if params:
                mlflow.log_params(params)
                logger.info(f"📌 Params: {params}")

            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"📊 {metric_name}: {score:.4f}")

            signature = infer_signature(input_data, model.predict(input_data))
            logger.info("🧠 Model signature inferred.")

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data.head(1),
                registered_model_name=model_name,
            )
            logger.info(f"✅ Model registered as: {model_name}")
            return model_info

    except Exception as e:
        logger.error(f"❌ MLflow logging failed: {e}")
        raise