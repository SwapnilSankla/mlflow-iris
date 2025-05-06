## MLFlow playground

- Project is generated using: `poetry new mlflow-playground/`
- Start MLFlow Tracking server: `poetry run mlflow ui`
- Train model: Run `poetry run python src/mlflow_iris/train_iris_model.py` from command line from `mlflow-playground` directory
- Invoke inference: Run `poetry run python src/mlflow_playground/inference_iris_model.py` from command line from `mlflow-playground` directory