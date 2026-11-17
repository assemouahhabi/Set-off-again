import argparse
import joblib
import yaml
import i18n
from pathlib import Path

from src.data_loader import load_data
from src.models import build_model, train_model
from src.explainers import build_explainer, compute_explanations

config_path = Path("config/experiment_01.yaml")
i18n.load_path.append(Path("i18n"))


def run_pipeline(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(i18n.t("FileNotFoundError", config_path=config_path))

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    # Load data
    X_train, X_test, y_train, y_test = load_data(config)

    # Check if the model is already trained and cached
    out_dir = Path(config.get("explainer", {}).get("out_dir", "output/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cache_path = out_dir / "trained_model.joblib"

    if model_cache_path.exists() and not config.get("force_retrain", False):
        # Load the cached model
        print(i18n.t("LoadingCachedModel", model_cache_path=model_cache_path))
        model = joblib.load(model_cache_path)
    else:
        # Build and train the model
        model = build_model(config)
        train_model(X_train, y_train, model, config)
        joblib.dump(model, model_cache_path)

    # Build the SHAP explainer based on the model type
    explainer = build_explainer(model, X_train, config)

    # Compute SHAP explanations for the test data
    shap_values = compute_explanations(model, X_test, explainer, config)

    exp_name = config.get("experiment_name", "unnamed_experiment")

    print(i18n.t("ExperimentCompleted", exp_name=exp_name))

    return model, shap_values, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SHAP experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    run_pipeline(Path("config/experiment_01.yaml"))
