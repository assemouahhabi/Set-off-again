from pathlib import Path
import shap
import numpy as np
import matplotlib.pyplot as plt


def _tree_explainer(model, X_test):
    return shap.TreeExplainer(model)


def _linear_explainer(model, X_test):
    return shap.LinearExplainer(model, X_test)


EXPLAINER_FACTORIES = {
    "random_forest": _tree_explainer,
    "gradient_boosting": _tree_explainer,
    "xgboost": _tree_explainer,
    "logistic_regression": _linear_explainer,
}


def build_explainer(model: object, X_test: np.ndarray, config: dict):
    """
    Build a SHAP explainer based on the model type specified in the configuration.
    Args:
        model: Trained machine learning model.
        X_test (ndarray): Testing features.
        config (dict): Configuration dictionary containing parameters for explanation computation.
    Returns:
        explainer: SHAP explainer object.
    """
    model_config = config.get("model", {})
    model_type = model_config.get("type", "random_forest")
    if model_type in EXPLAINER_FACTORIES:
        return EXPLAINER_FACTORIES[model_type](model, X_test)
    else:
        raise ValueError(f"Unsupported model type for SHAP explanations: {model_type}")


def compute_explanations(model: object, X_test: np.ndarray, config: dict):
    """
    Compute SHAP explanations for the provided model and test data.
    Args:
        model: Trained machine learning model.
        X_test (ndarray): Testing features.
        config (dict): Configuration dictionary containing parameters for explanation computation.
    Returns:
        shap_values: SHAP values for the test data.
    """

    # Create a SHAP explainer based on the model type
    explainer = build_explainer(model, X_test, config)

    # Compute SHAP values for the test data
    shap_values = explainer.shap_values(X_test)

    out_dir = config.get("out_dir")
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save Raw Values
        np.save(out_dir / "shap_values.npy", shap_values)

        _generate_shap_plots(shap_values, X_test, out_dir)

    return shap_values


def _generate_shap_plots(shap_values, X_test, out_dir):
    """
    Plot a SHAP summary plot for the provided SHAP values and test data.
    Args:
        shap_values (ndarray): SHAP values for the test data.
        X_test (ndarray): Testing features.
        out_dir (Path): Directory to save the plots.
    """
    plt.figure()
    shap.summary_plot(shap_values, X_test)

    # Summary Dot Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
