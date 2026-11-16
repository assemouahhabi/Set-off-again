from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
import pandas as pd


def _breast_cancer_to_dataframe():
    """
    Load the breast cancer dataset and convert it to a pandas DataFrame.
    Returns:
        data (tuple): A tuple containing the features (X) and target labels (y) of the breast cancer dataset.
    """
    data = load_breast_cancer(as_frame=True, return_X_y=True)
    return data


def _load_diabetes_to_dataframe():
    """
    Load the diabetes dataset and convert it to a pandas DataFrame.
    Returns:
        data (tuple): A tuple containing the features (X) and target labels (y) of the diabetes dataset.
    """
    data = load_diabetes(as_frame=True, return_X_y=True)
    return data


def _load_wine_to_dataframe():
    """
    Load the wine dataset and convert it to a pandas DataFrame.
    Returns:
        data (tuple): A tuple containing the features (X) and target labels (y) of the wine dataset.
    """
    data = load_wine(as_frame=True, return_X_y=True)
    return data


DATASETS = {
    "breast_cancer": _breast_cancer_to_dataframe,
    "diabetes": _load_diabetes_to_dataframe,
    "wine": _load_wine_to_dataframe,
}


def build_data_loader(config: dict):
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset_name")
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}"
        )

    # Return the dataset loader function based on the dataset name
    return DATASETS[dataset_name]


def load_data(config):
    """
    Load the breast cancer dataset and split it into training and testing sets.
    Args:
        config (dict): Configuration dictionary containing parameters for data loading.
    Returns:
        X_train (ndarray): Training features.
        X_test (ndarray): Testing features.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
    """
    # Build the data loader based on the configuration
    data_loader = build_data_loader(config)

    X, y = data_loader()

    # Split the dataset into training and testing sets based on the configuration
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    if config.get("debug_mode", False):
        n = config.get("debug_sample_size", 100)  # Default debug sample size is 100
        X_train, X_test = X_train[:n], X_test[:n]
        y_train, y_test = y_train[:n], y_test[:n]
        pass

    return X_train, X_test, y_train, y_test
