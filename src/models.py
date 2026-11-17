import i18n
import numpy as np
from typing import Protocol


class TrainableModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...


def _random_forest(**params):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params)


def _gradient_boosting(**params):
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(**params)


def _logistic_regression(**params):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**params)


def _xgboost(**params):
    from xgboost import XGBClassifier

    return XGBClassifier(**params)


MODELS = {
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    "logistic_regression": _logistic_regression,
    "xgboost": _xgboost,
}


def build_model(config: dict):
    model_config = config.get("model")
    if not model_config or model_config["type"] not in MODELS:
        raise ValueError(i18n.t("InvalidModelConfig"))

    params = {k: v for k, v in model_config.items() if k != "type"}
    return MODELS[model_config["type"]](**params)


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, model: TrainableModel, config: dict
):
    """
    Train a machine learning model using the provided training data and configuration.
    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        config (dict): Configuration dictionary containing parameters for model training.
    Returns:
        model: Trained machine learning model.
    """
    # Train the model
    model.fit(X_train, y_train)

    return model
