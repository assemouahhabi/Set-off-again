import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression


def _random_forest(params):
    return RandomForestClassifier(**params)


def _gradient_boosting(params):
    return GradientBoostingClassifier(**params)


def _logistic_regression(params):
    return LogisticRegression(**params)


MODELS = {
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    "logistic_regression": _logistic_regression,
}


def build_model(config: dict):
    model_config = config.get("model")
    if not model_config or "type" not in MODELS:
        raise ValueError(
            f"Configuration must contain a 'model' dictionary with a 'type' key."
        )

    model_type = model_config["type"]
    if model_type not in MODELS:
        raise ValueError(f"Unknown model: {model_type}")

    params = {k: v for k, v in model_config.items() if k != "type"}
    return MODELS[model_config["type"]](**params)


def get_and_train_model(X_train: np.ndarray, y_train: np.ndarray, config: dict):
    """
    Train a machine learning model using the provided training data and configuration.
    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        config (dict): Configuration dictionary containing parameters for model training.
    Returns:
        model: Trained machine learning model.
    """

    model = build_model(config)

    # Train the model
    model.fit(X_train, y_train)

    return model
