from pathlib import Path


def compute_explanations(model, X_test, config):
    """
    Compute SHAP explanations for the provided model and test data.
    Args:
        model: Trained machine learning model.
        X_test (ndarray): Testing features.
        config (dict): Configuration dictionary containing parameters for explanation computation.
    Returns:
        shap_values: SHAP values for the test data.
    """
    import shap

    # Create a SHAP explainer based on the model type
    if config.get("model_type", "random_forest") == "random_forest":
        explainer = shap.TreeExplainer(model)
    elif config.get("model_type") == "logistic_regression":
        explainer = shap.LinearExplainer(model, X_test)
    elif config.get("model_type") == "xgboost":
        import xgboost as xgb

        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(
            f"Unsupported model type for SHAP explanations: {config.get('model_type')}"
        )

    # Compute SHAP values for the test data
    shap_values = explainer.shap_values(X_test)

    # Save SHAP values as a plot in out_dir if specified in the config
    out_dir = config.get("out_dir", None)
    if out_dir is not None:
        shap_values_path = Path(out_dir) / "shap_values.npy"

        np.save(shap_values_path, shap_values)

    return shap_values
