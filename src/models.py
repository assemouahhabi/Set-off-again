def get_and_train_model(X_train, y_train, config):
    """
    Train a machine learning model using the provided training data and configuration.
    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        config (dict): Configuration dictionary containing parameters for model training.
    Returns:
        model: Trained machine learning model.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    model_type = config.get(
        "model_type", "random_forest"
    )  # Default model type is Random Forest

    if model_type == "random_forest":
        n_estimators = config.get("n_estimators", 100)  # Default number of trees is 100
        max_depth = config.get(
            "max_depth", None
        )  # Default max depth is None (nodes are expanded until all leaves are pure)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=config.get("random_state", 42),
        )
    elif model_type == "logistic_regression":
        C = config.get("C", 1.0)  # Default regularization strength is 1.0
        model = LogisticRegression(C=C, random_state=config.get("random_state", 42))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)

    return model
