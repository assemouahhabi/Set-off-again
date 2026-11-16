import argparse
from logging import config
from pathlib import Path
import yaml
from src.data_loader import load_data
from src.models import get_and_train_model
from src.explainers import compute_explanations


def main():
    parser = argparse.ArgumentParser(description="Run a SHAP experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment_01.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    X_train, X_test, y_train, y_test = load_data(config)

    trained_model = get_and_train_model(X_train, y_train, config)

    compute_explanations(trained_model, X_test, config)

    exp_name = config.get("experiment_name", "unnamed_experiment")
    print(f"Experiment completed: {exp_name}")


if __name__ == "__main__":
    main()
