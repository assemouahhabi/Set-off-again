import argparse
from pathlib import Path
import yaml

breakpoint()

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

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    X_train, X_test, y_train, y_test = load_data(config)

    trained_model = get_and_train_model(X_train, y_train, config)

    compute_explanations(trained_model, X_test, config)

    print(f"Experiment completed: {config['experiment_name']}")


if __name__ == "__main__":
    main()
