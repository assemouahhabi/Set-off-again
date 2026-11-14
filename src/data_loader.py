from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


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
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    test_size = config.get("test_size", 0.2)  # Default test size is 20%
    random_state = config.get(
        "random_state", 42
    )  # Default random state for reproducibility

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if config.get("debug_mode", False):
        n = config.get("debug_sample_size", 100)  # Default debug sample size is 100
        X_train = X_train.iloc[:n]
        X_test = X_test.iloc[:n]
        y_train = y_train.iloc[:n]
        y_test = y_test.iloc[:n]
        pass

    return X_train, X_test, y_train, y_test
